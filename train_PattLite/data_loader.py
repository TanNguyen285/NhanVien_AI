import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight


class AdaptiveGrayRGBTransform:
  
    def __init__(self, image_size: int, augment: bool = False):
        base_ops = [
            transforms.Resize((image_size, image_size)),
        ]
        if augment:
            base_ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.IMAGENET
                ),
            ]
        base_ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        self.pipeline = transforms.Compose(base_ops)

    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.pipeline(img)


def make_loader(data_root: str, split: str, image_size: int,
                batch_size: int, augment: bool = False,
                write_log=print) -> DataLoader:
    folder = os.path.join(data_root, split)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục: {folder}")

    dataset = datasets.ImageFolder(
        root=folder,
        transform=AdaptiveGrayRGBTransform(image_size=image_size, augment=augment)
    )
    write_log(f"  [{split}] {len(dataset)} ảnh | Classes: {dataset.classes}")
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True)


def get_class_weights(train_loader: DataLoader, device: torch.device,
                      write_log=print) -> torch.Tensor:
    train_labels = [label for _, label in train_loader.dataset.samples]
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(train_labels),
                                         y=train_labels)
    write_log(f"⚖️  Class weights: {np.round(class_weights, 3)}")
    return torch.tensor(class_weights, dtype=torch.float).to(device)