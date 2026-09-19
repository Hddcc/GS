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
