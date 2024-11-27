import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from newtrain import train,test
from collections import OrderedDict
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from CNN import SimpleLeNet
from torchvision import datasets, transforms
import argparse
from torch.utils.data import DataLoader, Subset
def set_parameters(net, parameters: list[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
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

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader,opt,savepath=None):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.opt = opt
        self.path = savepath
    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        #train(self.net, self.trainloader, epochs=1)p
        train(self.opt,self.net,self.trainloader,savepath = self.path)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.opt,self.net, self.valloader)
        print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def start_client(client_data,opt):
    client = FlowerClient(client_data["model"], client_data["train_loader"], client_data["val_loader"],opt,savepath = "./checkpoint/"+str(client_data["model"])+".pth")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
     # data preparation
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=0, help="label")
    parser.add_argument("--manual_seed", type=int, default=1111, help="random seed setting")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=3, help="number of epochs to train")
    parser.add_argument("--valid_interval", type=int, default=1, help="validation interval")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    
    opt = parser.parse_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    valset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
     # 目标标签
    if opt.label == 147:
        subset = [1, 4, 7]   
        trainLoader,valloader = getDataLoder(opt,trainset,valset,[1,4,7])
    elif opt.label == 258:
        subset = [2, 5, 8]
        trainLoader,valloader = getDataLoder(opt,trainset,valset,[2,5,8])
    elif opt.label == 369:
        subset = [3, 6, 9]
        trainLoader,valloader = getDataLoder(opt,trainset,valset,[3,6,9])
    else:
        subset = [0,1,2,3,4,5,6,7,8,9]
        trainLoader,valloader = getDataLoder(opt,trainset,valset,[1,4,7,2,5,8,3,6,9])
    model = SimpleLeNet()
    client = FlowerClient(model, trainLoader,valloader,opt,savepath=None)
    fl.client.start_client(client=client.to_client(),server_address="127.0.0.1:8080" )
