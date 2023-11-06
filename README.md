<div align="center">
<h1> ViTMatte🐒</h1>
<h3> Boosting Image Matting with Pretrained Plain Vision Transformers</h3>

[Jingfeng Yao](https://github.com/JingfengYao)<sup>1</sup>, [Xinggang Wang](https://scholar.google.com/citations?user=qNCTLV0AAAAJ&hl=zh-CN)<sup>1 📧</sup>, [Shusheng Yang](https://github.com/vealocia)<sup>1</sup>, [Baoyuan Wang](https://sites.google.com/site/zjuwby/)<sup>2</sup>

<sup>1</sup> School of EIC, HUST, <sup>2</sup> Xiaobing.AI

(<sup>📧</sup>) corresponding author.

[![arxiv paper](https://img.shields.io/badge/arxiv-paper-orange)](https://arxiv.org/abs/2305.15272)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dc2qoJueNZQyrTU19sIcrPyRDmvuMTF3?usp=sharing)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![authors](https://img.shields.io/badge/by-hustvl-green)](https://github.com/hustvl)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitmatte-boosting-image-matting-with/image-matting-on-composition-1k-1)](https://paperswithcode.com/sota/image-matting-on-composition-1k-1?p=vitmatte-boosting-image-matting-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitmatte-boosting-image-matting-with/image-matting-on-distinctions-646)](https://paperswithcode.com/sota/image-matting-on-distinctions-646?p=vitmatte-boosting-image-matting-with)

</div>

#

## News
* **`Oct 19th, 2023`:** **ViTMatte has been accepted by [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253523004074) (IF=18.6)!**
* **`Sep 21th, 2023`:** **ViTMatte is now available in 🤗[HuggingFace Transformers](https://huggingface.co/docs/transformers/main/model_doc/vitmatte#vitmatte)!** Many thanks to [Niels](https://nielsrogge.github.io/)!
* **`June 12th, 2023`:** We released google colab demo.  Try ViTMatte online!
* **`June 9th, 2023`:**  Many thanks to [Lucas](https://scholar.google.com/citations?hl=zh-CN&user=p2gwhK4AAAAJ) for creating ViT and [twitting](https://twitter.com/giffmana/status/1667091401463537665) our ViTMatte paper!
* **`June 8th, 2023`:**  [Matte Anything](https://github.com/hustvl/Matte-Anything) is released! If you like ViTMatte, you may also like Matte Anything.

* **`May 27th, 2023`:**  We released pretrained weights of ViTMatte!

* **`May 25th, 2023`:**  We released codes of ViTMatte. The pretrained models will be coming soon!
* **`May 24th, 2023`:**  We released our paper on [arxiv](https://arxiv.org/abs/2305.15272). 

## Introduction
<div align="center"><h4>Plain Vision Transformer could also do image matting with simple ViTMatte framework!</h4></div>

![avatar](figs/vitmatte.png)

Recently, plain vision Transformers (ViTs) have shown impressive performance on various computer vision tasks, thanks to their strong modeling capacity and large-scale pretraining. However, they have not yet conquered the problem of image matting. We hypothesize that image matting could also be boosted by ViTs and present a new efficient and robust ViT-based matting system, named ViTMatte. Our method utilizes (i) a hybrid attention mechanism combined with a convolution neck to help ViTs achieve an excellent performance-computation trade-off in matting tasks. (ii) Additionally, we introduce the detail capture module, which just consists of simple lightweight convolutions to complement the detailed information required by matting. To the best of our knowledge, ViTMatte is the first work to unleash the potential of ViT on image matting with concise adaptation. It inherits many superior properties from ViT to matting, including various pretraining strategies, concise architecture design, and flexible inference strategies. We evaluate ViTMatte on Composition-1k and Distinctions-646, the most commonly used benchmark for image matting, our method achieves state-of-the-art performance and outperforms prior matting works by a large margin.

## Get Started

* [Installation](docs/installation.md)
* [Train](docs/train.md)
* [Test](docs/test.md)

## Demo

You could try to matting the demo image with its corresponding trimap by run:
```
python run_one_image.py \
    --model vitmatte-s \
    --checkpoint-dir path/to/checkpoint
```
The demo images will be saved in ``./demo``.
You could also try with your own image and trimap with the same file.

Besides, you can also try ViTMatte in [![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dc2qoJueNZQyrTU19sIcrPyRDmvuMTF3?usp=sharing). It is a simple demo to show the ability of ViTMatte.

## Results

Quantitative Results on [Composition-1k](https://paperswithcode.com/dataset/composition-1k)
| Model      | SAD   | MSE | Grad | Conn  | checkpoints |
| ---------- | ----- | --- | ---- | ----- | ----------- |
| ViTMatte-S | 21.46 | 3.3 | 7.24 | 16.21 | [GoogleDrive](https://drive.google.com/file/d/12VKhSwE_miF9lWQQCgK7mv83rJIls3Xe/view?usp=sharing) |
| ViTMatte-B | 20.33 | 3.0 | 6.74 | 14.78 | [GoogleDrive](https://drive.google.com/file/d/1mOO5MMU4kwhNX96AlfpwjAoMM4V5w3k-/view?usp=sharing) |

Quantitative Results on [Distinctions-646](https://paperswithcode.com/dataset/distinctions-646)
| Model      | SAD   | MSE | Grad | Conn  | checkpoints |
| ---------- | ----- | --- | ---- | ----- | ----------- |
| ViTMatte-S | 21.22 | 2.1 | 8.78 | 17.55 | [GoogleDrive](https://drive.google.com/file/d/18wIFlhFY9MPqyH0FGiB0PFk3Xp2xTHzx/view?usp=sharing) |
| ViTMatte-B | 17.05 | 1.5 | 7.03 | 12.95 | [GoogleDrive](https://drive.google.com/file/d/1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW/view?usp=sharing) |

## Citation
```
@article{vitmatte,
      title={ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers}, 
      author={Jingfeng Yao and Xinggang Wang and Shusheng Yang and Baoyuan Wang},
      journal={arXiv preprint arXiv:2305.15272},
      year={2023}
}
```
