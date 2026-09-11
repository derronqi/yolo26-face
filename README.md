# yolo26-face

基于 [ultralytics v8.4.143](https://github.com/ultralytics/ultralytics) 的 YOLO26 人脸检测 + 5 关键点（landmark）。

## WIDER FACE 结果

val 集，640 单尺度，标准协议（[widerface_evaluate](widerface_evaluate)）：

| Method       | Test Size | Easy  | Medium | Hard  |
| ------------ | --------- | ----- | ------ | ----- |
| yolo26n-face | 640       | 93.95 | 91.58  | **80.12** |
| yolo26s-face | 640       | 95.52 | 93.80  | **83.35** |
| yolo26m-face | 640       | 96.31 | 94.86  | **85.13** |

- **Hard 全面超过 yolov8-face 同档**（79.6 / 83.1 / 84.7），Easy/Medium 基本持平；
- 无 DFL（`reg_max=1`），推理无需分布展开，**对 CPU/NPU 等嵌入式部署友好**；
- 训练只使用**标准 WIDER FACE train**，无需额外数据。

## 模型权重

| 模型 | 参数量 | 权重下载 |
| --- | ----- | ------- |
| yolo26n-face | 2.80M | [Google Drive](待填) |
| yolo26s-face | 10.58M | [Google Drive](待填) |
| yolo26m-face | 23.54M | [Google Drive](待填) |

## 特性

- YOLO26 架构：无 DFL、E2E 双头（yolo26n 支持 NMS-free 推理）、MuSGD 优化器、ReLU 友好部署
- 5 关键点输出，可直接用于人脸对齐等下游任务
- vendored ultralytics：升级到 v8.4.143 并修复 torch 1.8.x 容器下的 DDP 验证兼容性（`all_gather_object`、`ReduceOp.AVG`）
- 导出适配：`Pose26` 支持 9 路原始输出（box/cls/kpts 分离），便于 RKNN/NPU 端解码

## 训练

```bash
# n 档（其他规模替换 yaml 即可）
python abl/train_abl.py --name yolo26n-face --device 0 --model yolo26n-face-e2e.yaml --weights yolo26n.pt --epochs 500
```

- 数据：标准 WIDER FACE train（YOLO pose 格式，5 关键点），`ultralytics/cfg/datasets/widerface.yaml`
- 配方：imgsz 640，batch 16，cos_lr，MuSGD（optimizer=auto），COCO 预训练权重起步
- 最佳 epoch 一般在 255~328，使用 `best.pt` 即可

## 评测

```bash
# 标准 NMS 协议（与 published 结果可比）
python test_widerface.py --weights runs/release/yolo26n-face/weights/best.pt
python widerface_evaluate/evaluation.py -p widerface_txt/xxx/
```

数据集准备与官方评测工具见 [widerface_evaluate](widerface_evaluate/README.md)。
