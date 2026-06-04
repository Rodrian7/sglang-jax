from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.managers.tp_worker import ModelWorker


@partial(jax.jit, static_argnames=["draft_n", "accept_width"],
         donate_argnames=["hidden_states"])
def _postprocess_and_gather(
    hidden_states, positions, accept_index, accept_length, predict, draft_n, accept_width
):
    accept_length = accept_length + 1
    accept_index_flat = accept_index.reshape(-1)
    mask = accept_index_flat >= 0
    safe_pred_idx = jnp.where(mask, accept_index_flat, 0)
    verified_id = jnp.where(mask, predict.ravel()[safe_pred_idx], 0)
    n = accept_index_flat.shape[0]
    req_ids = jnp.arange(n) // accept_width
    per_req_last = req_ids * draft_n + draft_n - 1
    safe_index = jnp.where(accept_index_flat >= 0, accept_index_flat, per_req_last)
    return hidden_states[safe_index, :], positions[safe_index], verified_id, accept_length


@partial(jax.jit, static_argnames=["draft_n", "accept_width"],
         donate_argnames=["hidden_states", "logits"])
def _postprocess_and_gather_with_logits(
    logits, hidden_states, positions, accept_index, accept_length, predict, draft_n, accept_width
):
    accept_length = accept_length + 1
    accept_index_flat = accept_index.reshape(-1)
    mask = accept_index_flat >= 0
    safe_pred_idx = jnp.where(mask, accept_index_flat, 0)
    verified_id = jnp.where(mask, predict.ravel()[safe_pred_idx], 0)
    n = accept_index_flat.shape[0]
    req_ids = jnp.arange(n) // accept_width
    per_req_last = req_ids * draft_n + draft_n - 1
    safe_index = jnp.where(accept_index_flat >= 0, accept_index_flat, per_req_last)
    return logits[safe_index, :], hidden_states[safe_index, :], positions[safe_index], verified_id, accept_length


def replicate_to_mesh(
    mesh: jax.sharding.Mesh, *arrs: jax.Array
) -> tuple[jax.Array, ...] | jax.Array:
    """Replicate arrays across a mesh under explicit sharding.

    JIT outputs are typically vocab/data-sharded; spec-decode host orchestration
    (top_k, gather, build_tree) needs replicated arrays.
    """
    out = jax.device_put(arrs, NamedSharding(mesh, P()))
    return out[0] if len(out) == 1 else out


def replicate_to_mesh_jit(
    mesh: jax.sharding.Mesh, *arrs: jax.Array
) -> tuple[jax.Array, ...] | jax.Array:
    """JIT-safe replicate — uses jax.sharding.reshard instead of device_put."""
    sharding = NamedSharding(mesh, P())
    out = tuple(jax.sharding.reshard(a, sharding) for a in arrs)
    return out[0] if len(out) == 1 else out


class BaseDraftWorker(ABC):
    """Draft model worker interface for speculative decoding.

    Concrete implementations hold the draft model runner and own all
    draft-specific logic (multi-step decode, tree building, extend).
    Standard EAGLE uses ``EagleDraftWorker``; MTP uses
    ``MultiLayerDraftWorker``.
    """

    @property
    @abstractmethod
    def draft_model_runner(self):
        """Primary model runner (multi-runner workers return a designated one)."""

    @abstractmethod
    def draft(self, model_worker_batch):
        pass

    @abstractmethod
    def draft_extend_for_prefill(self, model_worker_batch, hidden_states, next_token_ids):
        pass

    @abstractmethod
    def draft_extend_for_decode(self, model_worker_batch, batch_output):
        pass


class BaseSpecWorker:
    """Speculative decode orchestrator.

    Owns a ``target_worker`` (the full model) and a ``draft_worker``
    (the draft/MTP model). The orchestration loop (prefill → draft →
    verify → draft_extend) and ``verify()`` itself are spec-algorithm-
    agnostic, so they live here; subclasses only differ in which
    ``BaseDraftWorker`` they construct.
    """

    def __init__(self, server_args, target_worker: ModelWorker, draft_worker: BaseDraftWorker):
        self.server_args = server_args
        self._target_worker = target_worker
        self._draft_worker = draft_worker

        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.mesh = target_worker.mesh

        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()

    @property
    def target_worker(self) -> ModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> BaseDraftWorker:
        return self._draft_worker

    # -- Main entry point --

    def forward_batch_speculative_generation(self, model_worker_batch: ModelWorkerBatch):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        if model_worker_batch.forward_mode.is_extend():
            if model_worker_batch.sampling_info.temperatures.ndim == 1:
                model_worker_batch.sampling_info.temperatures = (
                    model_worker_batch.sampling_info.temperatures[:, None]
                )
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
                vocab_size=self.target_worker.model_config.vocab_size,
            )
            logits_output, next_token_ids, cache_miss_count, bid, _seq_lens = (
                self.forward_target_extend(model_worker_batch, sampling_metadata)
            )
            if model_worker_batch.dp_size > 1:
                from jax.experimental.multihost_utils import process_allgather

                next_token_ids = process_allgather(next_token_ids, tiled=True)
            self.draft_worker.draft_extend_for_prefill(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=model_worker_batch.spec_info_padded,
                allocate_lens=np.asarray(model_worker_batch.seq_lens)[
                    model_worker_batch.logits_indices_selector
                ],
                bid=bid,
                cache_miss_count=cache_miss_count,
                extend_input_len_per_req=None,
                extend_logprob_start_len_per_req=None,
            )

        # spec_info.allocate_lens is DP-padded (total_bs,) at dp>1; gather back
        # to global-flat (real_bs,) so reqs_info[0].spec_info stays flat-ordered.
        sel = model_worker_batch.logits_indices_selector
        cur_allocate_lens = np.asarray(model_worker_batch.spec_info_padded.allocate_lens)[sel]
        self.draft_worker.draft(model_worker_batch)
        batch_output = self.verify_and_extend(model_worker_batch, cur_allocate_lens)
        return batch_output

    def forward_target_extend(self, model_worker_batch: ModelWorkerBatch, sampling_metadata):
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids, cache_miss_count = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, sampling_metadata=sampling_metadata
            )
        )
        return (
            logits_output,
            next_token_ids,
            cache_miss_count,
            model_worker_batch.bid,
            model_worker_batch.seq_lens,
        )

    def verify_and_extend(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens: jax.Array):
        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
        from sgl_jax.srt.layers.logits_processor import LogitsMetadata
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode, ForwardBatch, ForwardMode,
            get_global_expert_location_metadata,
        )
        from sgl_jax.srt.speculative.eagle_draft_worker import topk_probs_from_logits
        from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
        from sgl_jax.srt.utils.jax_utils import device_array

        spec_info: EagleVerifyInput = model_worker_batch.spec_info_padded
        spec_info.allocate_lens = cur_allocate_lens
        spec_info.prepare_for_verify(model_worker_batch, self.page_size, self.target_worker)

        target_mr = self.target_worker.model_runner
        draft_mr = self.draft_worker.draft_model_runner
        target_attn = target_mr.attn_backend
        draft_attn = draft_mr.attn_backend
        data_sharding = NamedSharding(target_mr.mesh, P("data"))
        sel = model_worker_batch.logits_indices_selector
        sel_np = np.asarray(sel)
        bs = len(model_worker_batch.seq_lens)
        draft_n = self.speculative_num_draft_tokens
        accept_width = self.speculative_num_steps + 1
        dp = model_worker_batch.dp_size
        per_dp_bs = model_worker_batch.per_dp_bs_size if dp > 1 else bs
        expert_loc = get_global_expert_location_metadata()

        # === Phase 1a: tree-independent metadata (overlaps with build_tree TPU) ===

        cm = spec_info.custom_mask
        if cm is not None:
            if hasattr(cm, 'dtype') and cm.dtype == jnp.bool:
                cm = cm.astype(jnp.int32)
            cm.copy_to_host_async()

        verify_meta = target_attn.compute_verify_metadata_numpy(
            cache_loc=model_worker_batch.cache_loc,
            seq_lens=model_worker_batch.seq_lens,
            draft_token_num=spec_info.draft_token_num,
            logits_indices_selector=sel_np,
            dp_size=dp,
            per_dp_bs=per_dp_bs,
        )
        has_swa = len(verify_meta) > 7
        if has_swa:
            (verify_pi, verify_ext_sl, verify_cu_q, verify_cu_kv,
             verify_seq_lens, aligned_seq_lens, verify_dist, swa_pi) = verify_meta
        else:
            (verify_pi, verify_ext_sl, verify_cu_q, verify_cu_kv,
             verify_seq_lens, aligned_seq_lens, verify_dist) = verify_meta

        dext_seq_lens = model_worker_batch.seq_lens.copy()
        dext_seq_lens[sel_np] += draft_n - 1
        dext_extend_seq_lens = np.zeros(bs, dtype=np.int32)
        dext_extend_seq_lens[sel_np] = accept_width
        dext_logits_indices = (
            dext_extend_seq_lens.reshape(dp, per_dp_bs)
            .cumsum(axis=1, dtype=np.int32).ravel() - 1
        )

        dext_pi, dext_cu_q, dext_cu_kv, dext_sl, dext_dist = (
            draft_attn.compute_dext_metadata_numpy(
                cache_loc=model_worker_batch.cache_loc,
                dext_seq_lens=dext_seq_lens,
                extend_seq_lens=dext_extend_seq_lens,
                allocate_lens=np.asarray(cur_allocate_lens),
                logits_indices_selector=sel_np,
                dp_size=dp,
                per_dp_bs=per_dp_bs,
            )
        )

        # === Phase 1b: H2D upload numpy-only arrays (still overlaps with build_tree) ===
        numpy_arrays = [
            model_worker_batch.seq_lens, model_worker_batch.out_cache_loc,
            model_worker_batch.req_pool_indices, model_worker_batch.cache_loc,
            model_worker_batch.extend_prefix_lens,
            verify_pi, verify_ext_sl, verify_cu_q, verify_cu_kv,
            verify_seq_lens, verify_dist,
            dext_pi, dext_cu_q, dext_cu_kv, dext_sl, dext_dist,
            dext_extend_seq_lens, dext_logits_indices, dext_seq_lens,
        ]
        if has_swa:
            numpy_arrays.append(swa_pi)

        uploaded = device_array(numpy_arrays, sharding=data_sharding)

        idx = 0
        seq_lens_dev = uploaded[idx]; idx += 1
        out_cache_loc = uploaded[idx]; idx += 1
        req_pool_indices = uploaded[idx]; idx += 1
        cache_loc = uploaded[idx]; idx += 1
        extend_prefix_lens = uploaded[idx]; idx += 1
        verify_pi_dev = uploaded[idx]; idx += 1
        verify_ext_sl_dev = uploaded[idx]; idx += 1
        verify_cu_q_dev = uploaded[idx]; idx += 1
        verify_cu_kv_dev = uploaded[idx]; idx += 1
        verify_sl_dev = uploaded[idx]; idx += 1
        verify_dist_dev = uploaded[idx]; idx += 1
        dext_pi_dev = uploaded[idx]; idx += 1
        dext_cu_q_dev = uploaded[idx]; idx += 1
        dext_cu_kv_dev = uploaded[idx]; idx += 1
        dext_sl_dev = uploaded[idx]; idx += 1
        dext_dist_dev = uploaded[idx]; idx += 1
        dext_extend_seq_lens_dev = uploaded[idx]; idx += 1
        dext_logits_indices_dev = uploaded[idx]; idx += 1
        dext_seq_lens_dev = uploaded[idx]; idx += 1
        swa_pi_dev = uploaded[idx] if has_swa else None

        # === Phase 1c: custom_mask repack (blocks on build_tree completion) ===
        custom_mask_dev = None
        if cm is not None:
            cm_host = np.asarray(cm)
            packed = target_attn.repack_custom_mask_numpy(
                cm_host, verify_seq_lens, aligned_seq_lens,
                spec_info.draft_token_num, dp, per_dp_bs,
            )
            custom_mask_dev = device_array(packed, sharding=data_sharding)

        # === Phase 2: build verify metadata + ForwardBatch ===
        verify_forward_metadata = FlashAttentionMetadata()
        verify_forward_metadata.cu_q_lens = verify_cu_q_dev
        verify_forward_metadata.cu_kv_lens = verify_cu_kv_dev
        verify_forward_metadata.page_indices = verify_pi_dev
        verify_forward_metadata.seq_lens = verify_sl_dev
        verify_forward_metadata.distribution = verify_dist_dev
        verify_forward_metadata.custom_mask = custom_mask_dev
        if swa_pi_dev is not None:
            verify_forward_metadata.swa_page_indices = swa_pi_dev

        input_ids_dev = jax.device_put(spec_info.draft_token, data_sharding)
        positions_dev = jax.device_put(spec_info.positions, data_sharding)

        # === Phase 3: dispatch verify chain ===
        verify_fb = ForwardBatch(
            bid=model_worker_batch.bid,
            forward_mode=model_worker_batch.forward_mode,
            batch_size=bs,
            input_ids=input_ids_dev, seq_lens=seq_lens_dev,
            out_cache_loc=out_cache_loc, positions=positions_dev,
            req_pool_indices=req_pool_indices, cache_loc=cache_loc,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=verify_ext_sl_dev,
            attn_backend=target_attn,
            spec_info=spec_info,
            spec_algorithm=model_worker_batch.spec_algorithm,
            capture_hidden_mode=model_worker_batch.capture_hidden_mode,
            expert_location_metadata=expert_loc,
        )
        model_worker_batch.forward_batch = verify_fb
        target_attn.forward_metadata = verify_forward_metadata
        verify_logits_metadata = LogitsMetadata.for_target_verify(model_worker_batch)

        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True,
            forward_metadata=verify_forward_metadata,
            logits_metadata=verify_logits_metadata,
        )

        is_all_greedy = model_worker_batch.sampling_info.is_all_greedy
        if is_all_greedy:
            logits_output.hidden_states = replicate_to_mesh(
                self.mesh, logits_output.hidden_states
            )
        else:
            logits_output.next_token_logits, logits_output.hidden_states = replicate_to_mesh(
                self.mesh, logits_output.next_token_logits, logits_output.hidden_states
            )
        spec_info.hidden_states = logits_output.hidden_states

        predict, accept_index, accept_length = spec_info.sample(
            model_worker_batch, logits_output,
            self.draft_worker.draft_model_runner.rngs, self.mesh,
        )

        positions_gather = replicate_to_mesh(self.mesh, spec_info.positions)

        if is_all_greedy:
            gathered_hs, dext_positions, verified_id, accept_length = (
                _postprocess_and_gather(
                    logits_output.hidden_states, positions_gather,
                    accept_index, accept_length, predict,
                    draft_n=draft_n, accept_width=accept_width,
                )
            )
        else:
            logits_output.next_token_logits, gathered_hs, dext_positions, verified_id, accept_length = (
                _postprocess_and_gather_with_logits(
                    logits_output.next_token_logits,
                    logits_output.hidden_states,
                    positions_gather,
                    accept_index, accept_length, predict,
                    draft_n=draft_n, accept_width=accept_width,
                )
            )

        # === Phase 4: dispatch dext immediately (no device_get!) ===
        dext_forward_metadata = FlashAttentionMetadata()
        dext_forward_metadata.cu_q_lens = dext_cu_q_dev
        dext_forward_metadata.cu_kv_lens = dext_cu_kv_dev
        dext_forward_metadata.page_indices = dext_pi_dev
        dext_forward_metadata.seq_lens = dext_sl_dev
        dext_forward_metadata.distribution = dext_dist_dev
        dext_forward_metadata.custom_mask = None
        draft_attn.forward_metadata = dext_forward_metadata

        dext_spec_info = EagleDraftInput(
            hidden_states=gathered_hs,
            accept_length=accept_length,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        dext_fb = ForwardBatch(
            bid=model_worker_batch.bid,
            forward_mode=ForwardMode.DRAFT_EXTEND,
            batch_size=bs,
            input_ids=verified_id,
            seq_lens=dext_seq_lens_dev,
            out_cache_loc=out_cache_loc,
            positions=dext_positions,
            req_pool_indices=req_pool_indices,
            cache_loc=cache_loc,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=dext_extend_seq_lens_dev,
            attn_backend=draft_attn,
            spec_info=dext_spec_info,
            spec_algorithm=model_worker_batch.spec_algorithm,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            expert_location_metadata=expert_loc,
        )

        dext_logits_metadata = LogitsMetadata.for_draft_extend(
            extend_seq_lens=dext_extend_seq_lens_dev,
            logits_indices=dext_logits_indices_dev,
            accept_lens=accept_length,
        )

        draft_logits_output, _, _ = draft_mr.forward(
            dext_fb, logits_metadata=dext_logits_metadata,
        )

        topk_p, topk_index = topk_probs_from_logits(
            draft_logits_output.next_token_logits, self.topk,
        )
        rep_hidden = replicate_to_mesh_jit(
            self.mesh, draft_logits_output.hidden_states,
        )

        # === Phase 5: device_get — all TPU work done ===
        jax.copy_to_host_async(topk_p)
        jax.copy_to_host_async(topk_index)
        jax.copy_to_host_async(rep_hidden)
        if hasattr(verified_id, "copy_to_host_async"):
            jax.copy_to_host_async(verified_id)

        predict_host, accept_length_host = jax.device_get((predict, accept_length))
        predict_host = np.asarray(predict_host)
        accept_length_host = np.asarray(accept_length_host)

        select_index = sel_np * accept_width + accept_length_host[sel_np] - 1
        topk_p_host = np.asarray(topk_p)[sel_np]
        topk_index_host = np.asarray(topk_index)[sel_np]
        hidden_host = np.asarray(rep_hidden)[select_index]
        verified_id_host = np.asarray(verified_id)[select_index]

        new_seq_lens = model_worker_batch.seq_lens + accept_length_host
        next_draft_input = EagleDraftInput(
            verified_id=verified_id_host,
            hidden_states=hidden_host,
            topk_p=topk_p_host,
            topk_index=topk_index_host,
            new_seq_lens=new_seq_lens,
            allocate_lens=cur_allocate_lens,
        )

        model_worker_batch.spec_info_padded = next_draft_input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict_host,
            next_draft_input=next_draft_input,
            accept_lens=accept_length_host,
            allocate_lens=cur_allocate_lens,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )
