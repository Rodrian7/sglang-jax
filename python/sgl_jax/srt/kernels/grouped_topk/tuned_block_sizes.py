"""Tuned `(block_tokens, unroll)` for `grouped_topk_pallas`, keyed by device + routing shape.

Tuned jointly by `benchmark/kernels/grouped_topk/tune_grouped_topk_bt.py` on real TPU: BT trades
grid-step overhead, `unroll` (the final-select `fori_loop` factor, 1..topk) trades pick-overlap
against the largest BT that fits VMEM. Lookup returns None on a miss so callers use a safe default.

Key: (next_power_of_2(T_local), E, G, Gtop, k), T_local = per-device token count. Self-contained
(only `jax`) to keep the kernel embeddable.
"""

import logging

import jax

logger = logging.getLogger(__name__)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _device_name() -> str:
    """Normalized TPU name, mirrors sgl_jax.srt.utils.jax_utils.get_device_name (e.g. 'TPU v7')."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("not a TPU device")
    if kind.endswith(" lite"):
        return kind[: -len(" lite")] + "e"
    if kind == "TPU7x":
        return "TPU v7"
    if kind and kind[-1] in ("e", "p"):
        return kind  # already e.g. "TPU v6e" / "TPU v5p"
    return kind


# device_name -> {(pow2(T), E, G, Gtop, k): (BT, unroll)}
# unroll=8 below = full unroll (topk) -- the pre-lever default; re-run the tuner to refine it.
TUNED_BT: dict[str, dict[tuple, tuple[int, int]]] = {
    "TPU v7": {
        # E=256 (DeepSeek-V3 / Ling)
        (64, 256, 8, 4, 8): (64, 8),
        (128, 256, 8, 4, 8): (128, 8),
        (256, 256, 8, 4, 8): (256, 8),
        (512, 256, 8, 4, 8): (512, 8),
        (1024, 256, 8, 4, 8): (1024, 8),
        (2048, 256, 8, 4, 8): (2048, 8),
        (4096, 256, 8, 4, 8): (2048, 8),  # BT=4096 exceeds v7x scoped VMEM with padded outputs
        (8192, 256, 8, 4, 8): (2048, 8),  # multi-block double-buffer caps BT at 2048
        (16384, 256, 8, 4, 8): (2048, 8),
        (32768, 256, 8, 4, 8): (2048, 8),
        # E=512 (MaxText)
        (64, 512, 8, 4, 8): (64, 8),
        (128, 512, 8, 4, 8): (128, 8),
        (256, 512, 8, 4, 8): (256, 8),
        (512, 512, 8, 4, 8): (512, 8),
        (1024, 512, 8, 4, 8): (1024, 8),
        (2048, 512, 8, 4, 8): (2048, 8),
        (4096, 512, 8, 4, 8): (2048, 8),  # E=512 doubles [BT,E] VMEM -> 4096 OOMs, cap 2048
        (8192, 512, 8, 4, 8): (2048, 8),
        (16384, 512, 8, 4, 8): (2048, 8),
        (32768, 512, 8, 4, 8): (2048, 8),
    },
}

_WARNED_MISSES: set[tuple] = set()


def get_tuned_config(T: int, E: int, G: int, Gtop: int, k: int) -> tuple[int, int] | None:
    """Tuned `(block_tokens, unroll)` for this device + key, or None on a non-TPU device / miss."""
    try:
        device = _device_name()
    except Exception:  # noqa: BLE001  (non-TPU device, etc.)
        return None
    key = (_next_power_of_2(T), E, G, Gtop, k)
    cfg = TUNED_BT.get(device, {}).get(key)
    if cfg is None and (device, key) not in _WARNED_MISSES:
        _WARNED_MISSES.add((device, key))
        logger.warning(
            "grouped_topk: no tuned (block_tokens, unroll) for device=%s key=%s; using default. "
            "Run benchmark/kernels/grouped_topk/tune_grouped_topk_bt.py to tune.",
            device,
            key,
        )
    return cfg


def get_tuned_bt(T: int, E: int, G: int, Gtop: int, k: int) -> int | None:
    """Tuned block_tokens only (BT-only view of `get_tuned_config`), or None on miss."""
    cfg = get_tuned_config(T, E, G, Gtop, k)
    return cfg[0] if cfg is not None else None
