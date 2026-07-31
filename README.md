# GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution
[![](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/tljxyys/GaussianSR) [![](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2407.18046) [![](https://img.shields.io/badge/Dataset-🔰DIV2K-blue.svg)](https://data.vision.ee.ethz.ch/cvl/DIV2K/) [![](https://img.shields.io/badge/Dataset-🔰test_data-blue.svg)](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) 
***
>**Abstract**: _Implicit neural representations (INRs) have revolutionized arbitrary-scale super-resolution (ASSR) by modeling images as continuous functions. However, existing INR-based ASSR methods discretely store latent codes, neglecting the continuous nature of image intensity variations and lacking interpretability of latent representations. This paper proposes a novel and elegant ASSR paradigm called GaussianSR that addresses these limitations through 2D Gaussian Splatting (2DGS). Instead of treating pixels as discrete points, GaussianSR models each pixel as a continuous Gaussian field. A classifier is trained to assign learnable Gaussian kernels with adaptive variances and opacities to each pixel, accommodating diverse input characteristics. By applying 2DGS to the encoder features, they are reorganized into a continuous field capturing inherent intensity variations. GaussianSR achieves flexible, adaptive receptive fields through mutually stacked Gaussian kernels, effectively capturing multi-scale features and long-range dependencies. Extensive experiments demonstrate that GaussianSR yields superior performance with reduced parameters, underscoring the great capability of our pipeline._
>
![image](https://github.com/tljxyys/GaussianSR/blob/main/fig/Figure_1.png)
***
## 1. Environment
- Python 3, Pytorch >= 1.6.0, TensorboardX, yaml, numpy, tqdm, imageio, einops

## 2. Running the code

**Train**: `python train_gaussian.py --config configs/train/train_edsr-baseline-gaussian-batch16.yaml`. We use 4 V100 GPUs for training EDSR-baseline-LIIF.

**Test**: `python test.py --config ./configs/test/test-div2k-2.yaml --model ./save/edsr-baseline-gaussiansr/epoch-last.pth` for div2k validation set, `python test.py --config ./configs/test/test-[benchmark]-2.yaml --model ./save/edsr-baseline-gaussiansr/epoch-last.pth` for benchmark datasets. We use a A100 GPU for testing.

## Citation
If you find the code helpful in your resarch or work, please cite our papers.
```
@misc{hu2024gaussiansrhighfidelity2d,
      title={GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution}, 
      author={Jintong Hu and Bin Xia and Bin Chen and Wenming Yang and Lei Zhang},
      year={2024},
      eprint={2407.18046},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.18046}, 
}
```
