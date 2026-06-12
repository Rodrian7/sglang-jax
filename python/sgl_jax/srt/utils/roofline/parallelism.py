"""Parallelism resolution + validation + theoretical collective-volume model.

The sglang-jax runtime builds a **2D ICI mesh** ``[data=dp_size, tensor=tp_size//
dp_size]`` whose total = ``tp_size`` devices (scheduler.py:296,
``create_device_mesh(ici_parallelism=[dp_size, tp_size//dp_size])``). So:

  * ``tp_size`` is the **mesh total = device count**, not the tensor-parallel
    degree;
  * the real **tensor-parallel degree** for attention / row+column-parallel
    linears is ``t = tp_size // dp_size`` (``attention_tp_size``,
    model_runner.py:78);
  * the **expert-parallel** group of the fused/fused_v2 MoE is the **full mesh**
    ``data * tensor = tp_size`` -- the ``--ep-size`` flag is ignored by the
    fused kernel (scheduler.py:301-312, fused_moe.py:138 shards experts over
    ``P(('data','tensor'), …)``).

`resolve()` takes the launch-style parallelism (the same numbers you pass to the
server: ``tp``, ``dp``, ``ep``, ``devices``) and returns the derived axis sizes,
raising ``ValueError`` on an inconsistent config so the roofline never silently
simulates an impossible layout. Volume helpers give the per-device theoretical
collective bytes (a balanced lower bound; real all-to-all is imbalance-bound).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelLayout:
    tp_total: int  # = device count = mesh(data*tensor)
    dp: int  # data axis
    t: int  # tensor axis = attention/linear TP degree = tp_total // dp
    ep: int  # effective expert-parallel group (fused MoE) = tp_total
    devices: int
    enable_sp: bool
    scatter_min: int  # tpu_scatter_min_local_size (default 128)
    moe_backend: str

    def should_scatter(self, tokens: int) -> bool:
        """SP reduce-scatter fires on the full mesh (data*tensor=tp_total) only
        above the per-device slice threshold and when it divides evenly."""
        D = self.tp_total
        if not self.enable_sp or D <= 1:
            return False
        return tokens >= D * self.scatter_min and tokens % D == 0


def _cfg(config, *names, default=None):
    for n in names:
        if config.get(n) is not None:
            return config[n]
    return default


def resolve(config, par, *, moe_backend="fused_v2", enable_sp=False, scatter_min=128):
    """Validate (tp, ep, dp, devices) against the model config + runtime mesh
    semantics; return a ``ParallelLayout``. Raises ``ValueError`` on violations,
    warns (returns warnings list) on the fused-EP override."""
    tp = int(par["tp"])
    dp = int(par.get("dp") or 1)
    devices = int(par.get("devices") or tp)
    ep_flag = par.get("ep")
    nh = _cfg(config, "num_attention_heads")
    n_exp = _cfg(config, "n_routed_experts", "num_experts")

    errors = []
    if devices != tp:
        errors.append(
            f"devices (={devices}) must equal tp_size (={tp}): the runtime mesh is "
            f"2D [data={dp}, tensor={tp}//{dp}] with total = tp_size devices "
            f"(scheduler.py:296). Pass --devices == --tp."
        )
    if dp <= 0 or tp % dp != 0:
        errors.append(
            f"tp_size (={tp}) must be divisible by dp_size (={dp}); the tensor-axis "
            f"size t = tp//dp must be a positive integer (scheduler.py:296)."
        )
    t = max(1, tp // max(1, dp))
    if nh is not None and nh % t != 0:
        errors.append(
            f"num_attention_heads (={nh}) must be divisible by attention_tp_size = "
            f"tp//dp (={t}) (model_config.py:635)."
        )
    if n_exp is not None and moe_backend in ("fused", "fused_v2") and n_exp % devices != 0:
        errors.append(
            f"n_routed_experts (={n_exp}) must be divisible by effective EP = "
            f"tp_size (={devices}) for moe_backend='{moe_backend}' (fused_moe.py:116)."
        )
    if errors:
        raise ValueError("Invalid parallelism for the runtime mesh:\n  - " + "\n  - ".join(errors))

    warnings = []
    ep_eff = devices  # fused MoE EP = full mesh
    if moe_backend in ("fused", "fused_v2") and ep_flag is not None and int(ep_flag) != ep_eff:
        warnings.append(
            f"moe_backend='{moe_backend}': effective EP = mesh(data*tensor) = tp_size "
            f"(={ep_eff}); the --ep-size (={ep_flag}) is ignored by the fused kernel "
            f"(scheduler.py:301-312). Using EP={ep_eff}."
        )

    return (
        ParallelLayout(
            tp_total=tp,
            dp=dp,
            t=t,
            ep=ep_eff,
            devices=devices,
            enable_sp=bool(enable_sp),
            scatter_min=int(scatter_min),
            moe_backend=moe_backend,
        ),
        warnings,
    )


def kv_heads_per_device(num_kv_heads: int, t: int) -> int:
    """Per-device KV heads under tensor-axis t, modelling REPLICATION: when
    t > num_kv_heads each device holds 1 head (KV replicated to t total), so it
    does NOT drop below 1 (model_config.get_total_num_kv_heads_with_replication)."""
    if t >= num_kv_heads:
        return 1
    return -(-num_kv_heads // t)  # ceil


# --------------------------------------------------------------------------
# Per-device theoretical collective volumes (bytes). Ring/bisection lower bound.
# --------------------------------------------------------------------------
def all_reduce_bytes(msg_bytes: float, p: int) -> int:
    """Ring all-reduce per-device traffic = 2*(p-1)/p * message."""
    if p <= 1:
        return 0
    return int(2.0 * (p - 1) / p * msg_bytes)


def reduce_scatter_bytes(msg_bytes: float, p: int) -> int:
    """Reduce-scatter (or all-gather) per-device traffic = (p-1)/p * message
    (half of all-reduce)."""
    if p <= 1:
        return 0
    return int((p - 1) / p * msg_bytes)


all_gather_bytes = reduce_scatter_bytes


def row_parallel_reduce_bytes(tokens: int, h: int, lp: ParallelLayout, *, dtype_bytes=2) -> int:
    """o_proj / MoE-output row-parallel reduce, per device.

    * SP (enable_sp and tokens clears the scatter threshold): reduce-scatter over
      the full mesh (tp_total) PLUS the residual all-gather to reshard back =
      2 * (D-1)/D * msg.
    * else (DP / below threshold): all-reduce over the tensor axis t =
      2*(t-1)/t * msg.
    """
    msg = tokens * h * dtype_bytes
    if lp.should_scatter(tokens):
        D = lp.tp_total
        return reduce_scatter_bytes(msg, D) + all_gather_bytes(msg, D)
    return all_reduce_bytes(msg, lp.t)
