"""BoolQ KLD adapter for PrivateLoRA HERA profiling.

For BoolQ, the Methods profiling term is the Kullback-Leibler divergence between
the pure-DCNM and single-layer ACIM-substituted {Yes, No} choice-token
distribution at the final answer position.
"""

from __future__ import annotations


def profile_boolq_choice_token_kld(*args, **kwargs) -> dict[str, float]:
    """Return per-PLM BoolQ KLD for single-layer ACIM substitution.

    Requires a user-provided PrivateLoRA-adapted Llama-2 checkpoint and the BoolQ
    evaluation set, which are not distributed here.  Compute the divergence with
    ``hera.affinity.profiling.compute_kld_from_logits`` over the {Yes, No}
    choice-token logits and merge with the EDP profile.
    """

    raise NotImplementedError(
        "Needs a user-supplied PrivateLoRA checkpoint and BoolQ data; "
        "use compute_kld_from_logits on your own choice-token logits."
    )

