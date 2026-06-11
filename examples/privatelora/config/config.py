"""Count-loss helpers used by the PrivateLoRA training path.

Sanitized from the original experiment config: only the functions imported by
``mymodels.modeling_llama_pl`` are kept.
"""

import numpy as np


def get_count(name, weight_module, logger):
    weight_q, weight_scale = weight_module.get_int_weight()
    weight_q_mapping = weight_q.reshape(32, 32, 9).permute(2, 1, 0).reshape(-1, 32)
    nonzero_indices = weight_q_mapping.nonzero()[:, 1]
    count_vector = np.bincount(nonzero_indices.cpu().numpy(), minlength=32)
    logger.info(f"{name} count num {round(np.mean((count_vector - 160) ** 2))} {count_vector}")


def get_count_loss(name, module):
    """Return (count_loss, mean column occupancy) over the INT weight matrix."""
    weight_q, weight_scale = module.get_int_weight()
    count_vector = weight_q.abs().sum(axis=0)
    count_loss = ((count_vector - 80) ** 2).mean()
    count_vector_mean = count_vector.mean()
    return count_loss, count_vector_mean
