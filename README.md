# Wanda + Rotation

```
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --rotate

```
```
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --rotate

```
