# yolocc 维护文档（给人和 AI）

> 最后更新：2026-05-03

### 1. 项目一句话说明

`yolocc` 是一个基于 YOLOv8 的可复用检测工程模板。当前默认任务是 4 类垃圾检测，但仓库结构已经整理成可直接迁移到其他桌面/家庭物体检测任务的通用工程。

项目已跑通完整闭环：真实数据下载 → auto-label → GPU 训练 → 评估 → 导出 ONNX → CLI/API 推理验证。不是 demo 骨架，是能继续迭代的真实工程。

## 2. 当前真实状态（2026-05-03）

### 数据集

当前训练配置主线已切到 `configs/template_object.yaml`。`configs/garbage.yaml` 仅作为旧 4 类垃圾示例任务保留。
为了兼容旧脚本，`data/dataset` 仍保留为 symlink → `data/real`，但 Windows 下不要依赖这个链接，直接用 `data/real` 更稳。

| Split | Images | Labels | Boxes |
|-------|--------|--------|-------|
| train | 1,766 | 1,766 | 1,867 |
| val | 377 | 377 | 394 |
| test | 384 | 384 | 428 |
| **total** | **2,527** | **2,527** | **2,689** |

来源：Stanford TrashNet（2,527 张真实垃圾照片），从 GitHub `garythung/trashnet` 仓库历史提交 `c536e5e` 提取 zip，再用预训练 YOLOv8n 自动生成 bounding box。

类别分布：

| Class | Boxes | 占比 |
|-------|-------|------|
| recyclable (0) | 2,252 | 84% |
| hazardous (1) | 53 | 2% |
| kitchen (2) | 109 | 4% |
| other (3) | 275 | 10% |

注意：TrashNet 原始数据只有 recyclable（cardboard/glass/metal/paper/plastic）和 trash（→other）。hazardous 和 kitchen 来自 auto-label 的 COCO 物体启发式映射（如 scissors→hazardous, banana→kitchen），数量少。

原始合成数据备份在 `data/dataset_synthetic_backup`。

### 类别定义

| ID | English | Chinese | 说明 |
|----|---------|---------|------|
| 0 | recyclable | 可回收物 | cardboard, glass, metal, paper, plastic |
| 1 | hazardous | 有害垃圾 | batteries, chemicals, medicine |
| 2 | kitchen | 厨余垃圾 | food waste, fruit peels |
| 3 | other | 其他垃圾 | cigarettes, ceramics, diapers |

**不能随便改类别 ID 顺序**，改了必须重新训练并同步所有配置/文档。

### 模型权重

| 文件 | 大小 | 说明 |
|------|------|------|
- `weights/best.pt` | 6.0 MB | 真实模型 PyTorch（30ep, RTX 4060）|
- `weights/best.onnx` | 12 MB | 真实模型 ONNX 640px |
- `weights/real_best.pt` | 6.0 MB | 同上原始文件 |
- `weights/real_best.onnx` | 12 MB | 同上原始文件 |
- `weights/smoke_best.pt` | 5.9 MB | smoke 模型（合成数据, 1ep CPU）|
- `weights/smoke_best.onnx` | 12 MB | smoke ONNX 320px |

**默认使用 `weights/best.pt` / `weights/best.onnx`**（真实模型）。

### 训练结果

| Metric | Test Set Value |
|--------|----------------|
| Precision | 0.488 |
| Recall | 0.465 |
| mAP50 | 0.436 |
| mAP50-95 | 0.382 |
| recyclable mAP50 | 0.831 |
| recyclable mAP50-95 | 0.766 |
| hazardous mAP50 | 0.345 |
| kitchen mAP50 | 0.190 |
| other mAP50 | 0.377 |
| Inference speed | 2.7ms/image (RTX 4060) |
| Model size | 6.0 MB (.pt) / 12 MB (.onnx) |

训练命令：
```bash
yolo detect train model=yolov8n.pt data=configs/garbage.yaml epochs=30 batch=16 imgsz=640 device=0 patience=15 project=results/runs/train name=real_detect exist_ok=True
```

### 测试

```bash
python -m pytest tests -q
# 71 passed in ~10s
```

| 测试文件 | 数量 | 覆盖内容 |
|----------|------|----------|
| test_config.py | 12 | Pydantic 配置校验、YAML 加载 |
| test_export.py | 4 | YAML/pyproject 元数据 |
| test_inference.py | 30 | Detection dataclass、backend 检测、后处理、绘制、letterbox 预处理、数据集生成/校验、CLI detect smoke、API imgsz 传递、ONNX auto imgsz、ONNX Runtime backend |
| test_api.py | 10 | FastAPI TestClient: /health、/info、/detect (json+图片+conf)、/detect/batch、OpenAPI schema |
| test_download_dataset.py | 15 | TrashNet 6→4 类别映射、YOLO 标签生成、COCO→垃圾分类映射、文件夹分类提取 |

### Git 状态

当前项目不在独立 git 仓库里。外层 `/mnt/c/Users/24560`（Windows `C:\Users\24560`）被初始化为 git 仓库且无提交。**不要执行 `git add -A`**，否则会把整个用户目录加进去。

如果要提交，已经在 `yolocc/` 目录单独初始化了仓库。

## 3. 目录结构

```text
yolocc/
├── .github/workflows/ci.yml    GitHub Actions CI（lint + test, Python 3.10/3.11/3.12）
├── .gitignore                   排除 __pycache__、*.pt、*.onnx、results/runs/ 等
├── configs/
│   ├── garbage.yaml             旧 4 类垃圾示例任务配置
│   ├── template_object.yaml     新版 6 类桌面/家庭检测模板配置
│   └── train_cfg.yaml           训练超参：model、epoch、batch、augmentation
├── data/
│   ├── dataset → data/real      symlink，当前指向真实数据
│   ├── dataset_synthetic_backup 合成数据备份（210+45+45 张几何图形）
│   ├── real/                    真实 TrashNet 数据（2,527 张）
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── .cache/                  下载缓存（trashnet zip/git 仓库）
├── deploy/
│   ├── app.py                   Streamlit 网页演示
│   └── opencv_deploy.py         OpenCV DNN 轻量部署脚本
├── docker/
│   ├── Dockerfile               多阶段构建（app + train）
│   └── docker-compose.yml       服务编排
├── scripts/
│   ├── download_real_dataset.py 真实数据下载/转换脚本（TrashNet + auto-label）
│   ├── prepare_dataset.py       合成数据生成、数据集划分、标签校验
│   └── run_pipeline.py          一键全流程（data → train → evaluate → export → benchmark）
├── src/
│   ├── __init__.py / __main__.py
│   ├── cli.py                   yolocc CLI 入口（Click）
│   ├── config.py                Pydantic 配置模型
│   ├── logger.py                Rich 日志
│   ├── train.py                 YOLOv8 训练入口
│   ├── validate.py              模型验证
│   ├── export_model.py          ONNX/TorchScript/OpenVINO/TFLite/NCNN 导出
│   ├── inference.py             统一推理引擎（.pt 用 Ultralytics，.onnx 默认优先 ONNX Runtime，失败回退 OpenCV DNN，支持 auto imgsz + letterbox）
│   ├── api.py                   FastAPI REST 服务（/health /info /detect /detect/batch）
│   ├── benchmark.py             推理速度 benchmark
│   └── visualize.py             训练曲线/类别分布/benchmark 图表
├── tests/
│   ├── test_config.py
│   ├── test_export.py
│   ├── test_inference.py
│   ├── test_api.py
│   └── test_download_dataset.py
├── weights/
│   ├── best.pt / best.onnx      真实模型（默认）
│   ├── real_best.pt / .onnx     真实模型原始文件
│   ├── smoke_best.pt / .onnx    smoke 模型（合成数据）
│   ├── model_card.md            模型训练详情/指标/限制
│   └── README.md                权重放置说明
├── results/
│   ├── demo_grid.jpg / demo_samples / demo_video.mp4  已有演示
│   ├── real_detect_test.jpg     真实推理结果示例
│   └── runs/                    训练输出（不提交 git）
├── Makefile                     常用命令快捷方式
├── pyproject.toml               包元数据、依赖、工具配置
├── requirements.txt             运行时依赖
├── README.md                    项目说明
└── MAINTENANCE.md               本文件
```

## 4. 环境安装

推荐 Python 3.10+。Windows 下直接在项目目录运行。

```bash
# 完整安装（开发 + API + UI）
pip install -e ".[all]"

# 或只装运行时
pip install -r requirements.txt
```

GPU 训练需要 NVIDIA GPU + CUDA。检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 5. 常用命令

**所有命令都在项目根目录下运行。Windows 用户先 `cd C:\Users\24560\Desktop\study\ccdemo\yolocc`。**

### 5.1 CLI 总览

```bash
python -m src.cli --help
```

子命令：train / validate / export / detect / benchmark / serve / data / viz

### 5.2 图片推理

```bash
# .pt 模型（GPU 推理，最快）
python -m src.cli detect --source data\dataset\images\test\0001.jpg --weights weights\best.pt --save results\my_test.jpg

# .onnx 模型（CPU 也能跑）
python -m src.cli detect --source data\dataset\images\test\0001.jpg --weights weights\best.onnx --save results\my_test.jpg --no-show
```

### 5.3 启动 API 服务

```bash
python -m src.cli serve --weights weights\best.onx --port 8000
```

然后：
- 浏览器打开 `http://localhost:8000/docs` 有 Swagger UI
- `curl http://localhost:8000/health`
- `curl -X POST http://localhost:8000/detect -F "file=@test.jpg"`

### 5.4 启动 Streamlit 网页演示

```bash
streamlit run deploy\app.py
```

浏览器打开 `http://localhost:8501`，左边选模型路径为 `weights\best.onx`，上传图片即可。

### 5.5 训练

```bash
# 标准训练（GPU）
yolo detect train model=yolov8n.pt data=configs/garbage.yaml epochs=100 batch=16 imgsz=640 device=0

# 快速 CPU 测试
yolo detect train model=yolov8n.pt data=configs/garbage.yaml epochs=1 batch=2 imgsz=320 device=cpu
```

### 5.6 验证

```bash
yolo val model=weights\best.pt data=configs\garbage.yaml split=test imgsz=640 device=0
```

### 5.7 导出 ONNX

```bash
yolo export model=weights\best.pt format=onnx imgsz=640 simplify=True
```

### 5.8 数据校验

```bash
python scripts\prepare_dataset.py validate --data-dir data\dataset
```

### 5.9 下载真实数据（需要网络）

```bash
# 下载 TrashNet + 转 YOLO 格式 + auto-label
python scripts\download_real_dataset.py --source trashnet --output data\real --auto-label
```

### 5.10 Makefile 快捷命令

```bash
make install        # pip install -e ".[all]"
make test           # pytest tests -q
make lint           # ruff check
make data           # 生成合成数据
make data-validate  # 校验数据
make train          # 训练
make export         # 导出 ONNX
make benchmark      # benchmark
make serve          # API 服务
make demo           # Streamlit
make pipeline       # 全流程
make docker-build   # Docker 构建
```

## 6. 数据集规范

YOLO detection 格式：

```
data/dataset/
  images/train/xxx.jpg    labels/train/xxx.txt
  images/val/xxx.jpg      labels/val/xxx.txt
  images/test/xxx.jpg     labels/test/xxx.txt
```

每行标签：`<class_id> <x_center> <y_center> <width> <height>`

- `class_id` 整数，必须 0-3
- 坐标归一化到 `[0,1]`
- `width/height` 必须 > 0
- 图片和标签同名

### 添加真实数据流程

1. 准备图片，按类别放在子文件夹里
2. 用 `scripts/download_real_dataset.py` 或手动标注
3. 确保 train/val/test 划分
4. `python scripts/prepare_dataset.py validate --data-dir data/你的目录`
5. 更新 symlink 或修改 `configs/garbage.yaml` 里的 path
6. 开始训练

## 7. 已完成的全部改进（按时间顺序）

### 第一轮：代码修复和工程基础

1. **修 CLI detect bug**：`Detection` dataclass 当 dict 访问 → 改为属性访问
2. **修 API 并发安全**：`detect_frame()` 增加单次 conf/iou 覆盖，finally 恢复
3. **修数据校验**：检查非法类别 ID、检查 bbox 宽高 > 0、CLI 返回非 0 exit code
4. **补依赖**：requirements.txt 补齐 click/rich/pydantic/requests/fastapi/uvicorn/python-multipart
5. **修 Dockerfile**：增加 `pip install -e .`
6. **修 Makefile benchmark**：权重路径修正
7. **README 重写**：去掉不存在的 CI/NCNN 文件声明，改为真实状态

### 第二轮：ONNX 自动 imgsz + API/推理修复

8. **ONNX auto imgsz**：`YOLODetector` 加载 ONNX 时自动从模型元数据读取输入尺寸
9. **letterbox 预处理**：OpenCV ONNX 推理改成标准 YOLO letterbox，避免非正方形图片被拉伸，正确还原坐标
10. **CLI serve --imgsz**：新增参数并传给 create_app
11. **tests/test_api.py**：9 个 FastAPI TestClient 测试
12. **weights/model_card.md**：训练细节/限制文档
13. **.gitignore**：排除大文件
14. **Pydantic response_model**：`src/api.py` 使用 `DetectResponse` / `BatchDetectResponse`，Swagger/OpenAPI 更清晰
15. **ONNX Runtime backend**：`YOLODetector(..., backend="onnxruntime")`，CLI `detect/serve` 支持 `--backend onnxruntime`，真实推理验证通过

### 第三轮：CI + 流水线 + Streamlit 修

13. **.github/workflows/ci.yml**：GitHub Actions CI（Python 3.10/3.11/3.12）
14. **run_pipeline.py**：提取硬编码权重路径为常量
15. **deploy/app.py**：Model Info 显示实际 imgsz

### 第四轮：真实数据 + 训练

16. **scripts/download_real_dataset.py**：TrashNet 下载 + 6→4 类别映射 + auto-label
17. **tests/test_download_dataset.py**：15 个测试
18. **真实训练**：TrashNet 2,527 张 → auto-label → 30 epochs RTX 4060 → mAP50=0.436
19. **导出**：ONNX 640px 12MB
20. **CLI 推理验证**：检测到 recyclable
21. **API 推理验证**：/health 200, /detect 200, 返回 detections
22. **pyproject.toml**：补齐 onnxsim/seaborn，展开 all extra 避免自引用

### 第五轮：工程级增强

23. **letterbox 预处理**：OpenCV ONNX 推理改成标准 YOLO letterbox + 正确坐标还原
24. **Windows 路径兼容**：`configs/garbage.yaml` 直接指向 `data/real`
25. **API schema 强类型**：`response_model` + OpenAPI 更清晰
26. **清理缓存**：`data/.cache` 从 41MB 清到 0
27. **ONNX Runtime backend**：CLI/API 可用 `--backend onnxruntime`，真实推理验证通过
28. **YOLOv8s 对比训练**：完成 30 epoch GPU 训练与 test 集评估，结论是当前数据条件下不如 `yolov8n`，因此 `best.*` 继续保留为 `yolov8n`

### 测试数量变化

`16 → 40 → 43 → 52 → 67 → 69 → 70 → 71 passed`

## 8. 已知问题 / 后续 TODO

### ✅ P0：闭环已完成

- 真实数据 + 真实训练 + 评估 + 导出 + CLI/API 推理全部通过
- 测试 71 passed

### P1：下一版定位（建议）

当前 4 类垃圾检测适合作为工程样板，但不适合作为“复杂场景高可靠识别”的最终版本。下一版建议切到更通用、也更贴近真实使用的任务定义：

- **推荐定位**：桌面/家庭物体检测模板
- **默认场景**：桌面杂物、日常物品、收纳辅助、危险物识别
- **保留能力**：训练 / 验证 / 导出 / API / Web / Docker / benchmark / tests
- **保留模型**：现有垃圾模型作为示例 checkpoint，不作为最终目标

为什么这样改：
1. 垃圾 4 类本身过于粗糙，复杂场景下很容易误检/漏检
2. auto-label 噪声大，继续堆训练轮数收益有限
3. 通用桌面/家庭场景更容易扩展类别，也更符合实际使用
4. 工程模板可以长期复用，后续换数据集就能继续迭代

### P2：下一步数据策略

- 先收集 100~300 张真实桌面/家庭场景图
- 只定义少量高价值类别（建议 5~8 个）
- 优先做手工精标，别再依赖 auto-label 作为主数据
- 保留现有训练脚本，但把默认数据集切成新任务
- 第一版 6 类草案与标注规范见 `DATA_GUIDE.md`

### P3：代码质量

- 补 Docker build 验证测试
- 可继续扩展 API schema（health/info 也用 response_model）
- OpenCV DNN 与 ONNX Runtime 做 benchmark 对比，并在 CLI 中增加 auto backend 策略

### P4：仓库整理

- 已经把 yolocc 目录初始化为独立 git 仓库
- 清理 data/.cache（下载缓存）
- 考虑用 Git LFS 管理大权重文件

## 9. 给后续 AI 的注意事项

1. **先读本文档**，再看 `README.md`、`configs/*.yaml`、`src/inference.py`
2. **不要改类别 ID 顺序**，改了必须重新训练并同步所有配置
3. **修改推理预处理时**，同时改 `src/inference.py` + `deploy/opencv_deploy.py` + 测试
4. **修改依赖时**，同时改 `pyproject.toml` + `requirements.txt` + `docker/Dockerfile`
5. **不要在当前 git 状态下执行 `git add -A`**，先在 yolocc 目录 `git init`
6. **每次修改后最低验证**：

```bash
python -m pytest tests -q
python -m src.cli detect --source data/dataset/images/test/0001.jpg --weights weights/best.pt --save /tmp/verify.jpg --no-show
```

7. **数据集路径**：`data/dataset` 是 symlink → `data/real`，Windows 下 symlink 可能不工作，如果报错就直接用 `data/real` 路径
8. **Windows 路径**：所有 `/` 在 Windows CMD 下用 `\`，PowerShell 两种都行
