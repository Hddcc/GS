# 尺度门控高频残差：首轮验证协议

## 版本与起点

- 分支：`experiment/face-scale-gated-frequency`。
- worktree：`GaussianSR-clean-scale-gated-frequency-20260918`。
- 起点：`7a5532d`，继承现有确定性评估与单卡梯度累积训练设施，不继承 GSASR 官方权重或 renderer 作为新模型。
- 保留 baseline 的 `models/gaussian.py` 与 `models/edsr.py`，未回滚、覆盖主工作区未提交的历史实验。
- baseline 中已有 oracle hook 默认关闭，新增模型不设置它；编码器保持 `edsr-baseline`。
- 使用服务器已有 seed-1 正式 baseline 最优 checkpoint，而不是五轮探针权重。程序检查同目录 `config.yaml` 的 `epoch_max >= 1350`；这证明配置来源，不能独立证明整个训练任务已完成，正式运行是否完成仍应由历史训练日志核实。

## 方法

- 模块一：沿用方向性频率特征编码和 query RGB 残差头，最终 RGB 层零初始化；不改变已有频率残差模型的定义或 checkpoint。
- 模块二：局部残差特征、相对 cell、四维连续尺度编码共同生成 sigmoid 标量门控。
- 公式：`prediction = baseline + 0.2 * gate * tanh(residual_head)`。
- 无尺度门控消融使用固定 `gate=0.5`；两组均为新模型的零初始化残差，不把旧频率残差正式结果当作本次消融。
- 所有倍率初始输出应与同一 baseline 一致，不针对某一倍率硬编码开启或关闭残差。
- 当前只验证两个模块，不额外改变边缘损失、编码器、Gaussian 几何或多 primitive。

## 固定训练与比较

- 三组均使用同一个 seed-1 正式 baseline checkpoint，并记录文件 SHA-256、来源 epoch；Adam 状态重新初始化，不使用 `resume` 冒充同模型续训。
- `control`：只对原 baseline 做 30 轮微调，学习率 `1e-6`。
- `residual` / `gated`：前 3 轮冻结主干，只训练新模块；后 27 轮联合微调，主干 `1e-6`、新增模块 `1e-4`。
- 三组额外训练 epoch、数据量、effective batch、scale sampler、seed 相同；候选主干实际更新少 3 轮，因此这是实际训练策略对照，不是每个参数更新次数完全相同的纯结构对照。
- 两个残差组训练策略完全一致，可以直接检验新增尺度门控的增量价值。
- CelebA train HR，全数据；连续微调倍率 `[2,4]`，L1，无边缘加权。
- 微批次 4，累积 3，effective batch 12；各组独占一张物理卡，不使用 DataParallel 混合不同尺度。
- 默认 GPU 3 跑 control、GPU 4 跑 residual；两者完成后 GPU 4 跑 gated。只有一张卡时顺序运行。
- 验证仍为 x4，仅记录每轮指标；最后比较固定 epoch-last，不按多倍率结果挑 checkpoint。

## 首轮评测

- CelebA val 排序前 100 张，同图、同 resize、同 query 比较四个模型：原正式 baseline、同预算 control、residual、gated。
- 固定完整报告倍率 `2, 2.5, 3, 3.5, 4`，不只摘取提升的点。
- 沿用探针 PIL bicubic 降采样，HR 左上裁剪为 8 的倍数，LR 尺寸 round；JSON 保存真实 HR/LR 尺寸。非整数倍率是名义倍率，尺寸取整导致有效倍率有微小差异。
- 指标：float RGB PSNR、RGB clamp `[0,1]`、无边界 shave。与正式 Y-PSNR 数值不可直接比较。
- 输出每图 PSNR、每倍率均值、gated-control 和 gated-residual 的配对差值及图像 bootstrap 95% 区间。
- bootstrap 只描述图像抽样不确定性，不替代多 seed；机制诊断明确为最后一个 query chunk，不冒充整图统计。
- 如果特定固定倍率出现可信正信号，先由用户反馈决定是否在独立 CelebA test / Helen 与多 seed 确认，不自动扫描倍率、学习率、损失或更换模块。
- 现有 x8 负结果保留。本阶段目标是明确限定的 x2–x4 范围，不重写历史全倍率门禁。

## 本地验证与服务器交付

- CPU PyTorch 2.3：两种候选在 x2/x2.5/x3/x3.5/x4/x8 初始最大误差均为 0。
- 训练模式使用相同随机状态验证等价：原编码器的 Gumbel 采样必须对齐，不能直接比较连续两次随机前向。
- 已验证新增模块有效梯度、冻结主干不变、联合阶段主干更新、优化器恢复后的下一步一致、query 分块一致和错误 seed 拒绝。
- 当前电脑无 CUDA；服务器 CUDA smoke 必须通过才能启动，尚无真实人脸训练收益。
- transfer 包使用 ASCII 文件名，只包含运行所需 Python、shell、配置、校验清单与存在的许可证，不包含本 Markdown、数据集或权重。
- 包内独立 `runtime/` 运行；不会写入服务器原项目模型、训练脚本、已有 save/results，因此不再要求服务器历史 `models/__init__.py` 的 preimage hash。
- 启动锁防重复运行；已有 save 拒绝覆盖；所有脚本和校验清单使用 LF。
- 默认 checkpoint：`save/face_gaussian_baseline_seed1_formal_v2protocol/epoch-best.pth`，可用 `BASELINE_CHECKPOINT` 指定真实文件。

```bash
conda activate GS
unzip face_scale_gated_frequency_bundle_20260918.zip
PHYSICAL_GPUS=3,4 ADMISSION_GPU=4 EVALUATION_GPU=4 bash face_scale_gated_frequency_bundle_20260918/launch_face_scale_gated_frequency.sh
tail -f face_scale_gated_frequency_bundle_20260918/runtime/results/orchestrator.log
```

状态：实现与 CPU 机制验证完成；服务器训练、CUDA 准入与真实提升尚待反馈，未合并 main。

## 已交付单包

- 实现 commit：`00dfce19d8c61941b6b25fddeb098e9cd92c2a20`，已 push。
- 本地 ZIP：`D:/个人资料/项目/任意尺度/face_scale_gated_frequency_bundle_20260918.zip`。
- 大小：41,654 bytes，不含数据或训练权重，复用服务器正式 baseline。
- SHA-256：`ad6dae2d536b077682f0c211afdfc594a908e593f9746b708e8a59026a98f502`。
- 解压后的 23 项清单全通过，shell 语法和精简 runtime 独立 CPU 测试通过。
- baseline 默认路径不存在时程序停止，不自动降级到五轮权重。请用绝对路径设置 `BASELINE_CHECKPOINT` 指向正式 baseline 最优 checkpoint，并保留同目录 `config.yaml`。
- 如果找不到正式 checkpoint 或来源配置，请回传停止信息；不要重新从头训练或绕过正式来源检查。
