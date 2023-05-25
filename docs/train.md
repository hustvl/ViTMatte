## Train

### Training Dataset

You should prepare Adobe Image Matting Dataset and COCO for training.

* Get [Adobe Image Matting Dateset](https://sites.google.com/view/deepimagematting).
* For training, merge 'Adobe-licensed images' and 'Other' folder to use all 431 foregrounds and alphas.
* Get [COCO](https://sites.google.com/view/deepimagematting).

Check `ViTMatte/configs/common/dataloader` to modify PATH TO TRAINING DATA.

### Pretrained Model

* Get [DINO](https://github.com/facebookresearch/dino) pretrained ViT-S and [MAE](https://github.com/facebookresearch/mae) pretrained ViT-B.
* Or you could download and preprocess pretrained weights by

  ```
  cd ViTMatte/pretrained
  python preprocess.py
  ```

### Train ViTMatte

ViTMatte has 2 sizes: ViTMatte-S and ViTMatte-B.

You can modify configs in `configs/ViTMatte_S_100ep.py` and `configs/ViTMatte_B_100ep.py`.

Run:

```
    python main.py \
        --config-file configs/ViTMatte_S_100ep.py \
        --num-gpus 2
```

to train ViTMatte-S.

Run:

```
    python main.py \
        --config-file configs/ViTMatte_B_100ep.py \
        --num-gpus 2
```

to train ViTMatte-B.

Training output will be saved in `ViTMatte/output_of_train/`