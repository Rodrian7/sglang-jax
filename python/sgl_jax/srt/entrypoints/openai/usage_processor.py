from __future__ import annotations

from collections.abc import Mapping
from typing import Any, final

from sgl_jax.srt.entrypoints.openai.protocol import UsageInfo


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _details_if_cached(count: int) -> dict[str, int] | None:
        """Return {"cached_tokens": N} only when N > 0 (keeps JSON slim)."""
        return {"cached_tokens": count} if count > 0 else None

    @staticmethod
    def _aggregate_spec_details(
        responses: list[dict[str, Any]],
    ) -> dict[str, float] | None:
        total_verify = 0
        total_accepted = 0
        for r in responses:
            mi = r["meta_info"]
            total_verify += mi.get("spec_verify_ct", 0)
            total_accepted += mi.get("spec_accepted_tokens", 0)
        if total_verify == 0:
            return None
        details = {
            "spec_verify_ct": total_verify,
            "spec_accepted_tokens": total_accepted,
            "spec_accept_length": total_accepted / total_verify,
        }
        first_ratio = next(
            (r["meta_info"].get("spec_accept_ratio") for r in responses
             if "spec_accept_ratio" in r["meta_info"]),
            None,
        )
        if first_ratio is not None:
            draft_tokens = round((total_accepted - total_verify) / (total_verify * first_ratio)) if first_ratio > 0 else 0
            if draft_tokens > 0:
                details["spec_accept_ratio"] = (total_accepted - total_verify) / (total_verify * draft_tokens)
        return details

    @staticmethod
    def calculate_response_usage(
        responses: list[dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        completion_tokens = sum(r["meta_info"]["completion_tokens"] for r in responses)

        prompt_tokens = sum(
            responses[i]["meta_info"]["prompt_tokens"] for i in range(0, len(responses), n_choices)
        )

        cached_details = None
        if enable_cache_report:
            cached_total = sum(r["meta_info"].get("cached_tokens", 0) for r in responses)
            cached_details = UsageProcessor._details_if_cached(cached_total)

        spec_details = UsageProcessor._aggregate_spec_details(responses)

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
            spec_details=spec_details,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        spec_details: dict[str, float] | None = None,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0)
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = (
            UsageProcessor._details_if_cached(sum(cached_tokens.values()))
            if enable_cache_report
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
            spec_details=spec_details,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: dict[str, int] | None = None,
        spec_details: dict[str, float] | None = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=cached_tokens,
            completion_tokens_details=spec_details,
        )
