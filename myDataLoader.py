import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class FilteredMNIST(Dataset):
    def __init__(self, original_dataset, target_labels=None):
        self.original_dataset = original_dataset
        self.target_labels = target_labels
        # 过滤出指定标签的数据
        if target_labels is None:
            self.filtered_data = [(img, label) for img, label in zip(original_dataset.data, original_dataset.targets)]
            label_counts = {label: 0 for label in range(10)}
        else:
            self.filtered_data = [
                (img, label) for img, label in zip(original_dataset.data, original_dataset.targets)
                if label in self.target_labels
            ]
            label_counts = {label: 0 for label in self.target_labels}
        
        for _, label in self.filtered_data:
            label_counts[label.item()] += 1
        print("Filtered label counts:", label_counts)
    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, index):
        img, label = self.filtered_data[index]
        # img = img.unsqueeze(0).float()  # 使图像有单通道（1通道，灰度图）
        # img = transforms.ToTensor()(img)  # 转换为 Tensor
        return img, label
# if __name__ == "__main__":
#     pass
    # 加载 MNIST 数据集
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # # 加载完整的 MNIST 数据集
    # trainset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # valset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # # 只保留数字 1, 4, 7
    # target_labels = [1, 4, 7]

    # # 过滤训练集和验证集
    # trainset_filtered = FilteredMNIST(trainset, target_labels)
    # valset_filtered = FilteredMNIST(valset, target_labels)
    # valset_filtered = FilteredMNIST(valset)
    # # 使用 DataLoader
 #   trainloader = DataLoader(trainset_filtered, batch_size=10, shuffle=True)
 #   valloader = DataLoader(valset_filtered, batch_size=opt.batch_size, shuffle=False)
