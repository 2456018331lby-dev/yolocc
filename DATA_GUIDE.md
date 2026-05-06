# Desktop/Home Object Detection Data Guide

## 1. 目标

为 `configs/template_object.yaml` 准备第一版真实数据集，用于把 yolocc 从“垃圾检测示例”推进到“通用桌面/家庭物体检测模板”。

第一阶段目标不是追求特别多类别，而是先做一个：
- 类别少
- 边界清晰
- 容易采集
- 容易明显提升效果

的数据集。

## 2. 推荐 6 类

对应 `configs/template_object.yaml`：

1. `container`
   - 例子：瓶子、杯子、易拉罐、保温杯
   - 不包含：纸盒、书本

2. `trash`
   - 例子：包装袋、纸团、餐巾纸、零食包装、一次性垃圾
   - 不包含：完整可用纸张/书本

3. `electronics`
   - 例子：数据线、充电器、遥控器、耳机盒、小电子配件
   - 不包含：大件显示器、整台电脑

4. `hazard`
   - 例子：剪刀、小刀、打火机、尖锐金属件
   - 目标是“需要特别注意的物体”

5. `paper`
   - 例子：书、本子、A4 纸、便利贴、文件
   - 不包含：揉皱纸团（那种应归 `trash`）

6. `misc`
   - 例子：暂时不属于前面 5 类、但在桌面/家庭中常见的可见物
   - 注意：不要把 `misc` 用成“懒得分就都丢这里”

## 3. 第一阶段采集建议

建议先做 100~300 张图片，不求大，但求稳：

- 真实桌面
- 不同光照
- 不同背景
- 轻度遮挡
- 单物体 + 多物体混合
- 横屏竖屏都要有

比例建议：
- container: 20%
- trash: 20%
- electronics: 20%
- hazard: 10%
- paper: 20%
- misc: 10%

重点：
- `hazard` 数量虽然可以少，但必须有代表性
- `trash` 和 `paper` 要故意做“相似但不同”的样本
- `electronics` 要避免都长得一模一样

## 4. 标注规则

### 4.1 框什么
- 框可见主体的最小外接矩形
- 框尽量贴边，不要留太大空白
- 部分遮挡也要框，只要人眼还能识别

### 4.2 不框什么
- 完全看不清类别的目标
- 极小到几乎无法学习的噪点
- 反光导致只剩一块白斑的目标

### 4.3 容易混淆的地方
- 纸张完整可读 → `paper`
- 揉成团、明显废弃包装 → `trash`
- 剪刀 / 小刀 / 打火机 → `hazard`
- 数据线 / 充电头 / 遥控器 → `electronics`
- 瓶子 / 杯子 / 罐子 → `container`

## 5. 文件组织

建议最终数据结构：

```text
data/template/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

对应配置文件：
- `configs/template_object.yaml`

## 6. 第一版训练建议

等第一批真实数据准备好后：

```bash
yolocc train --cfg configs/train_cfg.yaml --data configs/template_object.yaml --epochs 50 --device 0 --name object_detect_v1
```

然后验证：

```bash
yolocc validate --weights results/runs/train/object_detect_v1/weights/best.pt --data configs/template_object.yaml --device 0
```

## 7. 第一阶段成功标准

不是追求论文指标，而是追求“用户肉眼明显觉得更靠谱”：

- 常见桌面目标能被识别出来
- 误把纸张当垃圾、误把危险物漏掉的情况显著减少
- 多物体场景比当前垃圾模型更稳定
- API / CLI / Web Demo 都能直接复用新模型
