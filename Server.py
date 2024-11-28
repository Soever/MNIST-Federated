import flwr as fl
import random
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation


import numpy as np
import torch
import multiprocessing

from myDataLoader import FilteredMNIST
from CNN import SimpleLeNet
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from newtrain import train,test
from client import FlowerClient,set_parameters,get_parameters
def evaluate(server_round: int,parameters,config) :
    DEVICE = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
    net = SimpleLeNet().to(DEVICE)
    valset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # task 1
    # val_indices = [i for i, label in enumerate(valset.targets) if label in [1,2,3,4,5,6,7,8,9]]
    # valset_filtered = Subset(valset, val_indices)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64)


    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

if __name__ == "__main__":
    # 启动Flower服务器，并监听指定的地址和端口
    strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 3 clients for training
    min_evaluate_clients=3,  # Never sample less than 3 clients for evaluation
    min_available_clients=3,  # Wait until all 3 clients are available
    evaluate_fn = evaluate,
        proximal_mu = 0.5,
    )
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    fl.server.start_server(
        server_address="127.0.0.1:8080",  # 监听所有 IP 地址上的 8080 端口
        config=fl.server.ServerConfig(num_rounds=500),  # 设置联邦学习的训练轮次
        strategy=strategy,
    )