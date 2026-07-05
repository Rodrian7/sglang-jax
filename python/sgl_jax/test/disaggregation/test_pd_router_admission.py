import argparse
import asyncio
from types import SimpleNamespace


def make_lb(**kwargs):
    from sgl_jax.srt.disaggregation.mini_lb import MiniLoadBalancer

    args = SimpleNamespace(
        host="127.0.0.1",
        port=30000,
        request_timeout_secs=10,
        prefill_urls=[("http://prefill", 31000)],
        decode_urls=["http://decode"],
        test_external_dp_routing=False,
        prefill_bootstrap_host=None,
        max_concurrent_requests=None,
        pd_decode_prealloc_soft_limit=kwargs.get("prealloc_limit", 0),
        pd_decode_oldest_prealloc_wait_ms_soft_limit=kwargs.get("wait_limit", 0.0),
        pd_router_admission_poll_ms=kwargs.get("poll_ms", 50),
        policy="random",
        pd_disaggregation=True,
    )
    return MiniLoadBalancer(args)


def test_router_args_parse_pd_decode_admission_limits():
    from sgl_jax.srt.disaggregation.router_args import RouterArgs

    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    args = parser.parse_args([
        "--pd-decode-prealloc-soft-limit",
        "8",
        "--pd-decode-oldest-prealloc-wait-ms-soft-limit",
        "5000",
        "--pd-router-admission-poll-ms",
        "25",
    ])

    router_args = RouterArgs.from_cli_args(args)

    assert router_args.pd_decode_prealloc_soft_limit == 8
    assert router_args.pd_decode_oldest_prealloc_wait_ms_soft_limit == 5000.0
    assert router_args.pd_router_admission_poll_ms == 25


def test_decode_admission_blocked_by_prealloc_queue():
    lb = make_lb(prealloc_limit=8)
    info = {
        "internal_states": [
            {"pd_decode_admission": {"prealloc_queue_size": 8}},
        ]
    }

    assert lb._decode_admission_blocked(info)


def test_decode_admission_blocked_by_oldest_wait():
    lb = make_lb(wait_limit=5000.0)
    info = {
        "internal_states": [
            {
                "pd_decode_admission": {
                    "prealloc_queue_size": 1,
                    "oldest_prealloc_wait_ms": 6000.0,
                }
            },
        ]
    }

    assert lb._decode_admission_blocked(info)


def test_decode_admission_allows_empty_backlog():
    lb = make_lb(prealloc_limit=8, wait_limit=5000.0)
    info = {
        "internal_states": [
            {
                "pd_decode_admission": {
                    "prealloc_queue_size": 0,
                    "oldest_prealloc_wait_ms": None,
                }
            },
        ]
    }

    assert not lb._decode_admission_blocked(info)


def test_wait_for_decode_admission_polls_until_unblocked(monkeypatch):
    import sgl_jax.srt.disaggregation.mini_lb as mini_lb

    lb = make_lb(prealloc_limit=8, poll_ms=25)
    infos = [
        {"internal_states": [{"pd_decode_admission": {"prealloc_queue_size": 9}}]},
        {"internal_states": [{"pd_decode_admission": {"prealloc_queue_size": 0}}]},
    ]
    calls = []
    sleeps = []

    async def fake_fetch_backend_json(session, decode_server, endpoint_candidates):
        calls.append((session, decode_server, endpoint_candidates))
        return infos.pop(0)

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(mini_lb, "fetch_backend_json", fake_fetch_backend_json)
    monkeypatch.setattr(mini_lb.asyncio, "sleep", fake_sleep)

    asyncio.run(lb._wait_for_decode_admission(object(), "http://decode"))

    assert [call[1] for call in calls] == ["http://decode", "http://decode"]
    assert [call[2] for call in calls] == [
        ("get_server_info", "server_info"),
        ("get_server_info", "server_info"),
    ]
    assert sleeps == [0.025]
