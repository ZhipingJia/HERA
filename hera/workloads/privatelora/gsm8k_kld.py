"""GSM8K KLD adapter for PrivateLoRA HERA profiling.

For GSM8K, the Methods profiling term is the Kullback-Leibler divergence between
the pure-DCNM and single-layer ACIM-substituted distributions at teacher-forced
answer-token positions.
"""

from __future__ import annotations


def profile_gsm8k_answer_token_kld(*args, **kwargs) -> dict[str, float]:
    """Return per-PLM GSM8K KLD for single-layer ACIM substitution."""

    raise NotImplementedError("To be released in the revised manuscript version")

