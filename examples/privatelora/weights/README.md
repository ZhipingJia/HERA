# PrivateLoRA Checkpoints

The full-precision PrivateLoRA checkpoints evaluated in the manuscript are
published as **GitHub Release assets** (each file is ~647 MB, too large for the
git tree):

https://github.com/ZhipingJia/HERA/releases/tag/v1.0.0

| Task | Asset | Accuracy (reproduced with `eval_pl.py`) | SHA-256 |
|---|---|---|---|
| BoolQ | [`pl_boolq_fp_r256.bin`](https://github.com/ZhipingJia/HERA/releases/download/v1.0.0/pl_boolq_fp_r256.bin) | 0.8862 (recorded best 0.8865) | `6369818fb6bcb572b0e72ed314df2d2f01d98a070d17b7095bc412fb6c305741` |
| GSM8K | [`pl_gsm8k_fp_r256.bin`](https://github.com/ZhipingJia/HERA/releases/download/v1.0.0/pl_gsm8k_fp_r256.bin) | 0.2729 (recorded best 0.2691) | `13cdde76f1b183af46a7802beb841577810637e62646683daf2f94f6f76f1b87` |

Verify after download:

```bash
sha256sum -c <<'EOF'
6369818fb6bcb572b0e72ed314df2d2f01d98a070d17b7095bc412fb6c305741  pl_boolq_fp_r256.bin
13cdde76f1b183af46a7802beb841577810637e62646683daf2f94f6f76f1b87  pl_gsm8k_fp_r256.bin
EOF
```

Each checkpoint contains the PrivateLoRA overlay only (no base-model weights):
the per-block `q/k/v` LoRA matrices including the 96 PLM `lora_mobile`
matrices (rank 256), plus the LM-head LoRA pair `lora_lm_head_A` /
`lora_lm_head_B` (Head LB, rank 4).

The frozen Llama-2-7b base model is **not** redistributed; download
`meta-llama/Llama-2-7b-hf` from the Hugging Face Hub (subject to the Llama 2
license) and point `PRIVATE_LLM_BASE_MODEL` at it. The PrivateLoRA adapters are
derivative fine-tuned components and are likewise provided for research use
subject to the Llama 2 Community License.
