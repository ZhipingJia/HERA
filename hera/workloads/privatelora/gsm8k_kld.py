"""GSM8K KLD adapter for PrivateLoRA HERA profiling.

For GSM8K, the Methods profiling term is the Kullback-Leibler divergence between
the pure-DCNM and single-layer ACIM-substituted distributions at teacher-forced
answer-token positions.
"""

from __future__ import annotations


def profile_gsm8k_answer_token_kld(*args, **kwargs) -> dict[str, float]:
    """Return per-PLM GSM8K KLD for single-layer ACIM substitution.

    Requires a user-provided PrivateLoRA-adapted Llama-2 checkpoint and the GSM8K
    evaluation set, which are not distributed here.  Compute the divergence with
    ``hera.affinity.profiling.compute_kld_from_logits`` over the teacher-forced
    answer-token logits and merge with the EDP profile.
    """

    raise NotImplementedError(
        "Needs a user-supplied PrivateLoRA checkpoint and GSM8K data; "
        "use compute_kld_from_logits on your own answer-token logits."
    )

