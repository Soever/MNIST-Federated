import random
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import time
import flwr as fl
import numpy as np
import torch
import multiprocessing
from torchvision import transforms
from train import train
from myDataLoader import FilteredMNIST
from CNN import SimpleLeNet
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from newtrain import train,test
from client import FlowerClient
def getDataLoder(opt,trainset,valset,subset):
    # 加载 MNIST 数据集
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
    return trainloader,valloader

def start_client(client_data,opt):
    client = FlowerClient(client_data["model"], client_data["train_loader"], client_data["val_loader"],opt,savepath = "./checkpoint/"+str(client_data["model"])+".pth")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_seed", type=int, default=1111, help="random seed setting")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--valid_interval", type=int, default=1, help="validation interval")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    
    opt = parser.parse_args()
    print("args", opt)

    # set seed7
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    # data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    valset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
     # 目标标签
    trainLoader147,valloader147 = getDataLoder(opt,trainset,valset,[1,4,7])
    trainLoader258,valloader258 = getDataLoder(opt,trainset,valset,[2,5,8])
    trainLoader369,valloader369 = getDataLoder(opt,trainset,valset,[3,6,9])
    model147 = SimpleLeNet()
    model258 = SimpleLeNet()
    model369 = SimpleLeNet()
    client147 = FlowerClient(model147, trainLoader147,valloader147,opt,savepath=None)
    client258 = FlowerClient(model258, trainLoader258,valloader258,opt,savepath=None)
    client369 = FlowerClient(model369, trainLoader369,valloader369,opt,savepath=None)
    clients = [
    {"model": model147, "train_loader": trainLoader147, "val_loader": valloader147},
    {"model": model258, "train_loader": trainLoader258, "val_loader": valloader258},
    {"model": model369, "train_loader": trainLoader369, "val_loader": valloader369},
    ]
    processes = []
    for client_data in clients:
        p = multiprocessing.Process(target=start_client, args=(client_data,opt))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()
    for p in processes:
        if p.is_alive():
            p.terminate() 
    # server = ServerApp(server_fn=server_fn)
    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    # backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # When running on GPU, assign an entire GPU for each client
    # if  torch.cuda.is_available():
        # backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Run simulation

    
    # train(opt,model147, trainLoader147,savepath = None)
    # loss, accuracy = test(opt,model147,valloader147)
    # print(loss, accuracy)
    # training
    # train(opt,model147,[1,4,7],savepath = "./checkpoint/mdoel147.pth")
    # train(opt,model258,[2,5,8],savepath = "./checkpoint/mdoel258.pth")
    # train(opt,model369,[3,6,9],savepath = "./checkpoint/mdoel369.pth")