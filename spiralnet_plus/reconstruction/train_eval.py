import time
import os
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device, numz):
    train_losses, test_losses = [], []

    train_loss_all, test_loss_all, train_acc_all, test_acc_all = [], [], [], []
    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss, acc_train = train(model, optimizer, train_loader, device,numz)
        t_duration = time.time() - t
        test_loss, acc_test = test(model, test_loader, device, numz)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)

        train_loss_all.append(train_loss)
        test_loss_all.append(test_loss)
        train_acc_all.append(acc_train)
        test_acc_all.append(acc_test)

        plt.figure()
        plt.plot(test_loss_all)
        plt.savefig('test_loss')
        plt.close()
        plt.figure()
        plt.plot(test_acc_all)
        plt.savefig('test_acc')
        plt.close()
        plt.figure()
        plt.plot(train_loss_all)
        plt.savefig('train_loss')
        plt.close()
        plt.figure()
        plt.plot(train_acc_all)
        plt.savefig('train_acc')
        plt.close()

def train(model, optimizer, loader, device,numz):
    model.train()

    total_loss,total_loss_r = 0,0
    acc_num = 0
    acc_g = 0
    total_num = 0
    for data in loader:
        optimizer.zero_grad()
        x = data.x.to(device)
        out = model(data)
        loss = F.l1_loss(out, x, reduction='mean') 
        loss.backward()
        total_loss += loss.item()
        optimizer.step()


    return total_loss / len(loader)



classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def test(model, loader, device, numz):
    model.eval()
    total_loss = 0
        
    with torch.no_grad():
        for i, data in enumerate(loader):
            pred = model(data)
            x = data.x.to(device)
            loss = F.l1_loss(pred, x, reduction='mean')
            total_loss += loss.item()
 
    return total_loss / len(loader)


