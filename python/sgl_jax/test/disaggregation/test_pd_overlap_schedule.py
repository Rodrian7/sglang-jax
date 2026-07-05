import threading
from types import SimpleNamespace


def test_dispatch_uses_overlap_loop_for_pd_prefill_and_decode():
    from sgl_jax.srt.managers.scheduler import dispatch_scheduler_event_loop

    calls = []

    class FakeScheduler:
        enable_overlap = True

        def event_loop_overlap_disagg_prefill(self):
            calls.append("prefill_overlap")

        def event_loop_normal_disagg_prefill(self):
            calls.append("prefill_normal")

        def event_loop_overlap_disagg_decode(self):
            calls.append("decode_overlap")

        def event_loop_normal_disagg_decode(self):
            calls.append("decode_normal")

        def event_loop_overlap(self):
            calls.append("null_overlap")

        def event_loop_normal(self):
            calls.append("null_normal")

    dispatch_scheduler_event_loop(
        FakeScheduler(),
        SimpleNamespace(disaggregation_mode="prefill"),
    )
    dispatch_scheduler_event_loop(
        FakeScheduler(),
        SimpleNamespace(disaggregation_mode="decode"),
    )

    assert calls == ["prefill_overlap", "decode_overlap"]


def test_prefill_chunk_resolves_overlap_result_before_handoff():
    from sgl_jax.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    calls = []
    launch_done = threading.Event()
    req = SimpleNamespace(bootstrap_room=1, rid="rid0", pd_time_stats=None)
    batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[req])],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )

    scheduler = SimpleNamespace(
        enable_overlap=True,
        tp_worker=SimpleNamespace(
            resolve_last_batch_result=lambda event=None: calls.append(("resolve", event))
        ),
        disagg_kv_manager=SimpleNamespace(use_raiden=True),
        chunked_reqs=[None],
        set_next_batch_sampling_info_done=lambda batch: calls.append(("sampling_done", batch)),
        _pd_mark_time=lambda req, name, **kwargs: calls.append(("mark", name)),
        _raiden_handoff_chunk=lambda req, req_id, is_final: calls.append(
            ("handoff", req_id, is_final)
        ),
    )

    SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
        scheduler,
        batch,
        SimpleNamespace(),
        launch_done,
    )

    assert calls[0] == ("resolve", launch_done)
    assert ("handoff", "rid0", True) in calls


def test_prefill_chunk_uses_batch_chunked_snapshot_for_final_flag():
    from sgl_jax.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    calls = []
    req = SimpleNamespace(bootstrap_room=1, rid="rid0", pd_time_stats=None)
    batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[req])],
        _pd_chunked_reqs=(),
    )
    scheduler = SimpleNamespace(
        enable_overlap=False,
        disagg_kv_manager=SimpleNamespace(use_raiden=True),
        # Simulate the global scheduler state already moving to the next batch.
        # The current batch snapshot must win.
        chunked_reqs=[req],
        set_next_batch_sampling_info_done=lambda batch: None,
        _pd_mark_time=lambda req, name, **kwargs: None,
        _pd_add_duration=lambda req, name, seconds: None,
        _raiden_handoff_chunk=lambda req, req_id, is_final: calls.append(is_final),
    )

    SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
        scheduler,
        batch,
        SimpleNamespace(),
    )

    assert calls == [True]


def test_prefill_chunk_snapshot_keeps_mid_chunk_when_global_state_advances():
    from sgl_jax.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    calls = []
    req = SimpleNamespace(bootstrap_room=1, rid="rid0", pd_time_stats=None)
    batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[req])],
        _pd_chunked_reqs=(req,),
    )
    scheduler = SimpleNamespace(
        enable_overlap=False,
        disagg_kv_manager=SimpleNamespace(use_raiden=True),
        chunked_reqs=[None],
        set_next_batch_sampling_info_done=lambda batch: None,
        _pd_mark_time=lambda req, name, **kwargs: None,
        _pd_add_duration=lambda req, name, seconds: None,
        _raiden_handoff_chunk=lambda req, req_id, is_final: calls.append(is_final),
    )

    SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
        scheduler,
        batch,
        SimpleNamespace(),
    )

    assert calls == [False]
