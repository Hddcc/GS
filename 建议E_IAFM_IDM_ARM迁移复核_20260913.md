# 建议 E：IAFMNet soft IDM + ARM 迁移复核

日期：2026-09-13  
来源：CVPR 2026《IAFMNet: Information-Aware Feature Modulation for Efficient Super-Resolution》  
实现状态：仅进入 GaussianSR seed-1 短程准入，不等同于完整 IAFMNet 复现。

## 1. 一手证据

IAFMNet 用信息密度图（IDM）指导两条路径：硬资源分配 IGRA 和软仿射重标定 ARM。本文只迁移 ARM，因为 Table 5 给出了受控的 soft-path 消融：Urban100 x2 从无 AR/无 IDM 的 `32.30 dB`，到 AR/无 IDM 的 `32.45 dB`，再到 AR+IDM 的 `32.52 dB`；Manga109 对应 `39.19 -> 39.26 -> 39.32 dB`。因此 soft-only 总增益为 `+0.22/+0.13 dB`，其中 IDM 相对无指导 AR 的独立增益为 `+0.07/+0.06 dB`。

Table 3 还显示 learned IDM 相对 Sobel guidance 在 Urban100/Manga109 x2 为 `+0.15/+0.18 dB`，支持“不重复固定 Sobel edge prior”的选择。论文最优信息熵权重为 `lambda=1e-4`；去掉 GDN 会在 Manga109 x2 降低 `0.21 dB`（`39.32 -> 39.11`），所以实现不能把 IDE 简化成普通卷积 attention。

补充材料 Fig.2 表明 IDM 在 iteration 0 接近随机，从 2k 到 10k iteration 才开始形成粗结构，到 100k--1M 才逐渐精细。因此本项目不使用 1 epoch 早停；5 epoch 对应约 7.5k optimizer steps，只足以检验短程粗结构和 PSNR 信号，不能声称完整收敛。

## 2. 与 GaussianSR 的最小迁移

- 保留原 EDSR head、16 个 ResBlock、body-tail、segmentation head、100 类 Gaussian bank、renderer 与 Fourier decoder。
- 从 shallow feature 通过两组 `3x3 Conv + GDN` 预测 Gaussian mean 与正 scale；训练时用 `U(-0.5,0.5)` 可微量化并按论文 Eq.3--4 计算离散 Gaussian likelihood 和信息熵。
- 为维持 A/B 公平，量化噪声采样后恢复每张 GPU 的 RNG state，使之后的 Gumbel-softmax 使用与 baseline 相同的随机流。
- 在第 4/8/12/16 个 EDSR residual block 后放置 ARM。每个 ARM 严格按 Eq.8--9 使用 `1x1` expansion、等分、`[feature, IDM]` 拼接生成 scale、另一半 depthwise `3x3` 局部特征以及逐元素相乘；以固定 residual scale `0.1` 接回原主干。
- 不迁移 IGRA、Top-k mask、submanifold sparse convolution、自注意力、PixelShuffle tail 或整套 IAFMNet。该实现是受控的 GaussianSR soft-path adaptation，不是论文架构复现。

## 3. 准入协议

- 只跑 seed 1、5 epoch、CelebA 原 bicubic 数据、L1、Adam `1e-4`、batch 12、倍率 `U(1.5,8)`；信息熵 auxiliary 固定 `1e-4`。
- 使用 physical GPU `0,2,4` 的 DataParallel，保持每卡 4、有效 batch 12；held-out 继续用同一 100 张 x2/x4/x8 RGB PSNR。
- 质量门禁：五轮相对 baseline 均值/末轮不低于 `-0.010/-0.020 dB`，held-out 三倍率平均严格为正，单倍率不低于 `-0.030 dB`。
- 机制门禁：信息熵、IDM spatial std、四个 ARM raw/scaled response 和 candidate-baseline output response 均大于 `1e-4`；CUDA smoke 还要求 IDE mean/scale、ARM 与原 encoder 梯度有限非零、GDN/ARM 方程误差不超过 `1e-7`、共享初始化误差为 0、评估严格可重复。
- 只有 seed 1 同时过质量与机制门禁才制作 seed 2/3。失败则停止该方向，不改 entropy weight、ARM 个数、residual scale 或阈值追结果。

## 4. 证据边界与许可

论文使用 DF2K、固定 x2/x3/x4、64x64 LR patch、batch 64、初始学习率 `1e-3` 和 1,000,000 iterations；本项目的数据、任务、decoder、训练长度和指标均不同，不能预先保证迁移收益。论文消融是单次报告，没有 seed 方差。

截至复核时 GitHub repository search 返回 0 个匹配官方仓库；因此没有复制第三方实现。代码仅依据公开方程独立实现，并明确保留“adaptation”限定。

## 5. 本地文献

- 主文：`D:\个人资料\项目\任意尺度\literature_cache_20260910_iafmnet\IAFMNet_CVPR_2026.pdf`；SHA-256 `8C24F419F62430140398C4145AF28AFE175F6589F3E0BDD3EA4D3218661F2614`。
- 补充材料：`D:\个人资料\项目\任意尺度\literature_cache_20260910_iafmnet\IAFMNet_CVPR_2026_supplemental_full.pdf`；SHA-256 `00075DB237C823F19D3D56B4C02788173A33F90A2538063EE53181DCD129563D`。
