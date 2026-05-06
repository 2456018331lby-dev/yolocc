# Quick commands for the new generic object-detection task

# 1) Put your labeled data into:
# data/template/images/{train,val,test}
# data/template/labels/{train,val,test}

# 2) Validate dataset structure
python scripts/prepare_dataset.py validate --data-dir data/template

# 3) Train first generic model
python -m src.cli train --cfg configs/train_cfg.yaml --data configs/template_object.yaml --epochs 50 --device 0 --name object_detect_v1

# 4) Validate trained model
python -m src.cli validate --weights results/runs/train/object_detect_v1/weights/best.pt --data configs/template_object.yaml --device 0

# 5) Export ONNX
python -m src.cli export --weights results/runs/train/object_detect_v1/weights/best.pt --format onnx --data configs/template_object.yaml

# 6) Run API with exported model
python -m src.cli serve --weights weights/best.onnx --backend onnxruntime --port 8000
