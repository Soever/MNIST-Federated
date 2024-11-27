import random
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
SAVE_MODEL_PATH = "./checkpoint/best_accuracy.pth"

def train(opt,model,subset=None,savepath = None):
    device = torch.device("cuda:0" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    valset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # 目标标签
    target_labels =subset

    # 找到训练集中符合条件的索引
    train_indices = [i for i, label in enumerate(trainset.targets) if label in target_labels]
    val_indices = [i for i, label in enumerate(valset.targets) if label in target_labels]

    # 使用 Subset 创建过滤后的数据集
    if subset is None:
        trainset_filtered = trainset
        valset_filtered = valset
    else:
        trainset_filtered = Subset(trainset, train_indices)
        valset_filtered = Subset(valset, val_indices)

    trainloader = torch.utils.data.DataLoader(trainset_filtered,batch_size=opt.batch_size)
    valloader = torch.utils.data.DataLoader(valset_filtered, batch_size=opt.batch_size)

    # training epoch loop
    best_eval_acc = 0
    start = time.time()
    for ep in range(opt.num_epoch):
        avg_loss = 0
        model.train()
        print(f"{ep + 1}/{opt.num_epoch} epoch start")

        # training mini batch
        for i, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.mitem()

            if i > 0 and i % 100 == 0:
                print(f"loss:{avg_loss / 100:.4f}")
                avg_loss = 0

        # validation
        if ep % opt.valid_interval == 0:
            tp, cnt = 0, 0
            model.eval()
            for i, (imgs, labels) in enumerate(valloader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    preds = model(imgs)
                preds = torch.argmax(preds, dim=1)
                tp += (preds == labels).sum().item()
                cnt += labels.shape[0]
            acc = tp / cnt
            print(f"eval accuracy:{acc:.4f}")

            # save best model
            if acc > best_eval_acc:
                best_eval_acc = acc
                savepath = SAVE_MODEL_PATH if savepath is None else savepath
                torch.save(model.state_dict(), savepath)
                print("saved best accuracy model")

        print(f"{ep + 1}/{opt.num_epoch} epoch finished. elapsed time:{time.time() - start:.1f} sec")

    print(f"training finished. best eval acc:{best_eval_acc:.4f}")









import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
# from flwr_datasets import FederatedDataset

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]