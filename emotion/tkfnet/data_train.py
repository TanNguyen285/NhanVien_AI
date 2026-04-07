import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_path, batch_size=128):
    # Data Augmentation cho tập Train (theo đúng thực tế huấn luyện FER)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform cho tập Test/Val (không xoay/lật)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Giả sử cấu trúc: data/train/ và data/test/
    train_dataset = datasets.ImageFolder(root=f"{data_path}/train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=f"{data_path}/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader