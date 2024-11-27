import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
def train(opt,net, trainloader, verbose=True,savepath = None):
    """Train the network on the training set."""
    device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=opt.lr)
    net.train()
    best_eval_acc = 0
    for epoch in range(opt.num_epoch):
        correct, total, epoch_loss = 0, 0, 0.0
        # for batch in trainloader:
        for i, (imgs, labels) in enumerate(trainloader):
            # images, labels = batch["img"].to(device), batch["label"].to(device)
            images, labels =imgs.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        # if epoch_acc > best_eval_acc:
        #     best_eval_acc = epoch_acc
        #     if savepath is not None:
        #         torch.save(net.state_dict(), savepath)
        #         print("saved best accuracy model")
def test(opt,net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(testloader):
            images, labels =imgs.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy