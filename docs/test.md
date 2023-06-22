## Test

### Test Dataset

Get preprocessed Composition-1k Dataset from [MatteFormer](https://github.com/webtoon/matteformer).

### Inference

Run

```
python inference.py \
    --config-dir path/to/config \
    --checkpoint-dir path/to/checkpoint \
    --inference-dir path/to/inference \
    --data-dir path/to/dataset
```

to infer on Composition-1k.

### Evaluation

NOTE:  The final quantitative results of ViTMatte is NOT evaluted by `evaluation.py`. We use official matlab code from [DIM](https://github.com/foamliu/Deep-Image-Matting) for fair comparision.

Run

```
python evaluation.py \
    --pred-dir path/to/inference \
    --label-dir path/to/composition_1k/alpha \
    --trimap-dir path/to/composition_1k/trimap \
```

to quick evaluate your inference results.
