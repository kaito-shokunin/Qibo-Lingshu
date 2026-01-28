import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import copy

# 定义数据集代理类以支持不同的转换（必须在全局作用域以支持 Windows 多线程）
class DatasetProxy(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def train_model():
    # 1. 配置路径和参数
    data_dir = r"E:\大创项目\知识图谱\Tongue coating classification 增强"
    save_dir = r"E:\大创项目\qibo_lingshu\models\tongue_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    # 针对 4060 GPU 优化的参数
    batch_size = 32  # 增加 Batch Size
    num_epochs = 50  # 显著增加训练轮数
    learning_rate = 0.0001 # 使用更小的学习率以获得更精细的权重调整
    num_classes = 6
    img_size = 256  # 提高分辨率
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. 高级数据增强 (使用 TrivialAugmentWide)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.TrivialAugmentWide(), # 自动选择增强策略
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), # 舌诊图像上下翻转有时也适用
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. 加载数据集
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")

    # 划分训练集和验证集 (85% / 15%)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data_raw, val_data_raw = random_split(full_dataset, [train_size, val_size], 
                                               generator=torch.Generator().manual_seed(42))
    
    train_dataset = DatasetProxy(train_data_raw, transform=data_transforms['train'])
    val_dataset = DatasetProxy(val_data_raw, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 使用更强大的 ResNet50 模型
    print("Initializing ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3), # 添加 Dropout 防止过拟合
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    # 5. 损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 使用标签平滑提高泛化能力
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 6. 训练循环
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_epoch = 0

    # 检查是否存在已有的模型并尝试加载（断点续训）
    model_path = os.path.join(save_dir, "tongue_model.pth")
    if os.path.exists(model_path):
        print(f"发现已有模型 {model_path}，正在尝试加载以继续训练...")
        try:
            checkpoint = torch.load(model_path)
            # 兼容不同格式的 checkpoint
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'accuracy' in checkpoint:
                    best_acc = checkpoint['accuracy']
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                print(f"成功加载模型，从 Epoch {start_epoch} 继续，之前最佳准确率: {best_acc:.4f}")
            else:
                # 只有模型权重的情况
                model.load_state_dict(checkpoint)
                print("加载了模型权重，将从头开始优化。")
            best_model_wts = copy.deepcopy(model.state_dict())
        except Exception as e:
            print(f"加载模型失败 ({str(e)})，将从头开始训练。")

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存表现最好的模型权重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 实时保存当前最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'classes': class_names
                }, os.path.join(save_dir, "tongue_model.pth"))
                print(f"Saved new best model with accuracy: {best_acc:.4f}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳权重并最终保存
    model.load_state_dict(best_model_wts)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': class_names,
        'model_type': 'resnet50'
    }, os.path.join(save_dir, "tongue_model.pth"))
    print(f"Final model saved to {os.path.join(save_dir, 'tongue_model.pth')}")

if __name__ == "__main__":
    train_model()
