# ResNet-50 Fine-Tuning on CIFAR-10

Fine-tuning a **ResNet-50** model pre-trained on ImageNet for image classification on the **CIFAR-10** dataset (10 classes).

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python main.py
```

The CIFAR-10 dataset is automatically downloaded to `.datasets/cifar10/`.

## Project Structure

```
├── main.py          # Entry point
├── model.py         # ResNet-50 model construction
├── dataset.py       # CIFAR-10 data loading and transforms
├── train.py         # Training and evaluation loop
└── requirements.txt # Dependencies 
```

## Technical Details

| Parameter  | Value                                      |
| ---------- | ------------------------------------------ |
| Base model | ResNet-50 (ImageNet weights)               |
| Dataset    | CIFAR-10 (60,000 32×32 images, 10 classes) |
| Optimizer  | SGD (lr = 1e-3)                            |
| Loss       | CrossEntropyLoss                           |
| Batch size | 64                                         |
| Epochs     | 30                                         |

### Adaptations for CIFAR-10

Since CIFAR-10 images are 32×32 (vs 224×224 for ImageNet), two modifications are applied to ResNet-50:

- **`conv1`**: replaced with a 3×3 kernel, stride 1, padding 1 (instead of 7×7, stride 2)
- **`maxpool`**: replaced with `Identity()`

## Classes

`plane` · `car` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`
