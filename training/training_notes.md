# Training Notes

## Dataset
- Dataset: Personal Protective Equipment - Combined Model
- Source: https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model
- License: CC BY 4.0
- Total images: 44,002
- Split: 30,765 train / 8,814 valid / 4,423 test
- Export format: YOLOv8

## Model
- Base model: YOLOv8n (COCO pretrained)
- Training image size: 640 x 640
- Batch size: 32
- Epochs: 50
- Early stopping patience: 15
- Optimizer: SGD
- Learning rate schedule: cosine
- Augmentation: Ultralytics default pipeline with mosaic, HSV, flip, and translate

## Hardware
- Training device: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- Training environment: local CUDA workstation
- Export formats: PyTorch .pt, ONNX FP32, ONNX INT8

## Results
- Test mAP@50: 0.776
- Test mAP@50-95: 0.491
- Precision: 0.704
- Recall: 0.812
- Per-class highlights: Hardhat 0.904, Person 0.946, NO-Hardhat 0.776, Safety Vest 0.658
- Validation peak mAP@50 from training log: 0.762

## Quantization Rationale
INT8 quantization was chosen to reduce the deployment footprint and keep the app responsive on CPU-only Hugging Face Spaces. The quantized model is much smaller than the FP32 export and supports real-time compliance monitoring without requiring a GPU. The small accuracy tradeoff is acceptable because the production target is fast, browser-based inference with graceful fallback behavior rather than offline batch evaluation.
