# 正式频率残差＋边缘损失逐图配对复核

## 目标与边界

- 分支 `experiment/face-formal-paired-audit`，从 `cb89371` 新建独立 worktree；不修改已有正式训练或新建质量模型。
- 复核已有 1,350 epoch 的 seed-1 正式 Baseline 与“方向高频查询残差＋边缘加权 L1”候选；不重新推理，不训练，不调整模型或损失。
- 主关注范围固定为正式八倍率中已测试的 `2,2.5,3.5,4`。原正式测试没有 x3 逐图数据，不凭空增加倍率；完整报告原有 `1.5,2,2.5,3.5,4,5.5,7.5,8`，不能事后只留下胜出的点。
- 两个数据集分别检验：CelebA test1000、Helen test50。指标是同协议 Y-PSNR、Y-SSIM、RGB LPIPS-Alex；不混用最近门控探针的 CelebA val100 float RGB PSNR。
- 逐图比较先校验 dataset/method、倍率、裁边、有限指标、每倍率图片完整、唯一 `(scale,image)` 键和 Baseline/候选两侧完全相同的配对集合。
- 报告每尺度三个指标的 Baseline/候选均值、逐图配对平均差、改善图像占比，以及 5,000 次固定种子“以图为单位重采样”的 95% bootstrap 区间。
- 范围汇总先对同一图片的四个倍率取差值平均，再以图片为重采样单位；不能将 4 个倍率视作相互独立样本。LPIPS 低为优，改善占比按负差值计算。
- bootstrap 只衡量本批图片的抽样不确定性，不包含训练 seed 波动，也不将模型选择与验证数据复用造成的偏差归零。Helen 只有 50 张；不能由一个 seed 的图像 CI 宣称多 seed 显著性。
- 缺少/歧义 CSV 时停止并显示候选路径；可显式提供四个 CSV 参数。不覆盖任何已有 JSON、CSV、checkpoint 或日志。新报告是 `results/face_formal_v2protocol/paired_audit_20260919.json`。
- 当前本地无服务器原始 CSV，只有历史四位小数正式汇总；尚未计算真实 CI，也不能宣称该方向已通过此次复核。需服务器运行并反馈结果后决定是否做 seed 重复实验。

## 单包服务器运行

将包上传到 `/root/userfolder_new/20260527GaussiSR/GaussianSR-main` 后：

```bash
cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main
unzip face_formal_paired_audit_bundle_20260919.zip
bash face_formal_paired_audit_bundle_20260919/run_face_formal_paired_audit_20260919.sh
```

只读脚本使用标准库 Python，不需要 GPU。若自动定位报告文件名歧义，先检查：

```bash
find results/face_formal_v2protocol/metrics -type f -name '*per_image.csv'
```

然后用脚本提示的 `--baseline-celeba`、`--candidate-celeba`、`--baseline-helen`、`--candidate-helen` 四个参数指定路径。不要猜文件属于哪个模型；根据冻结运行日志确认来源。

## 交付状态

- 实现提交 `9b05e90d92135089cffefa6b536b2167b5fe5366`，已 push 到 `origin/experiment/face-formal-paired-audit`。
- 单包 `D:/个人资料/项目/任意尺度/face_formal_paired_audit_bundle_20260919.zip`，4,157 bytes，SHA-256 `345b8c3155ef635322a0ec3b5c653469fe8f03fa0e481574fb9ab4b40acb3091`。
- 解压后 `IMPLEMENTATION_COMMIT`、Python 脚本和 bash 脚本的清单校验均 OK；bash 语法、Python 编译及五项合成逐图 CSV 测试通过。
- 包内仅含 ASCII 命名的运行脚本、校验清单和实现提交，不含本 Markdown、原始 CSV、数据集、checkpoint 或任何训练改动。
- 首次真实数据分析待服务器反馈。若输出提示自动定位歧义，四条明确路径由服务器上正式实验冻结记录核对后指定；不得根据分析差值判断哪个文件应算候选。

## 真实复核结果（2026-09-19）

- CelebA test1000 / Helen test50 的逐图配对复核均完成，使用全部八个正式倍率；JSON：`results/face_formal_v2protocol/paired_audit_20260919.json`。
- 预先固定的 x2/x2.5/x3.5/x4 聚合差值：

| 数据集 | PSNR-Y | SSIM-Y | LPIPS |
|---|---:|---:|---:|
| CelebA | +0.004754 `[+0.000292,+0.009456]` | +0.000161 `[+0.000112,+0.000210]` | -0.000464 `[-0.000569,-0.000357]` |
| Helen | +0.027651 `[+0.003355,+0.053191]` | +0.000350 `[+0.000152,+0.000550]` | -0.000253 `[-0.000771,+0.000260]` |

- Helen x2.5/x3.5/x4 的 PSNR-Y 分别 `+0.057851/+0.043244/+0.037274 dB`；CelebA x4 为 `+0.009895 dB`。Helen x2 PSNR-Y `-0.027765 dB`，因此结论限定为中倍率和部分非整数倍率改善。
- 该结果确认正式“方向高频查询残差＋边缘加权 L1”是当前最适合论文的保底方案。它仍是 seed 1 结果，image bootstrap CI 不等同于多 seed 统计显著性。下一阶段应做冻结和可视化，而不是继续门控模块搜索。

## 冻结状态纠正（2026-09-19）

- 用户执行历史文档中的 `sha256sum -c results/face_formal_v2protocol/frozen/formal_manifest.sha256`，服务器返回 `No such file or directory`。盘点确认实际存在两个历史清单：`formal_baseline_manifest.sha256` 与 `formal_training_manifest.sha256`，以及两组正式 checkpoint、frozen 配置/日志、四份逐图 CSV 和配对复核 JSON；缺失的是文档中写错的汇总文件名，不是全部冻结资料缺失。
- 在两份实际清单验证完成前，仍不能声称正式 checkpoint、日志、代码和 CSV 已统一冻结。逐图配对 JSON 的真实性校验不受影响。
- 先运行下面的只读盘点命令，把实际存在的文件和目录发回；不要执行 `sha256sum ... > formal_manifest.sha256`，也不要覆盖 `results/face_formal_v2protocol`：

```bash
cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main
printf '%s\\n' '--- formal result tree ---'
find results/face_formal_v2protocol -maxdepth 3 -type f -printf '%p\\n' 2>/dev/null | sort
printf '%s\\n' '--- formal checkpoints/configs/logs ---'
find save/face_gaussian_baseline_seed1_formal_v2protocol \\
     save/face_gaussian_frequency_residual_edge_seed1_formal_v2protocol \\
     -maxdepth 1 -type f -printf '%p\\n' 2>/dev/null | sort
printf '%s\\n' '--- frozen directory ---'
ls -la results/face_formal_v2protocol/frozen 2>/dev/null || true
```

- 盘点后请验证实际存在的清单，而不是创建同名副本：

```bash
cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main
sha256sum -c results/face_formal_v2protocol/frozen/formal_baseline_manifest.sha256
sha256sum -c results/face_formal_v2protocol/frozen/formal_training_manifest.sha256
```

- 若某项 `FAILED`，先回传失败行和对应清单内容；不把当前工作树文件冒充正式训练时版本，也不删除或覆盖历史清单。
