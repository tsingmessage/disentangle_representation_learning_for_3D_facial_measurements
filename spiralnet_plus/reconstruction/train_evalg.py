import time
import os
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
import numpy as np
import seaborn as sns
import umap
import matplotlib.pyplot as plt

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device, numz):
    train_losses, test_losses = [], []

    train_loss_all, test_loss_all, train_acc_all, test_acc_all = [], [], [], []
    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss, acc_train = train(model, optimizer, train_loader, device)
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

def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    acc_num = 0
    total_num = 0
    for data in loader:
        optimizer.zero_grad()
        y = data.y.to(device).float()
        x = data.x.to(device)
        out, gender, pz, z = model(data)
        criterion = torch.nn.BCELoss()
        loss = F.l1_loss(out, x, reduction='mean') + 0.15*criterion(gender, y) + 0.6*F.mse_loss(z, pz)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        for i in range(len(gender)):
            total_num = total_num + 1
            dis = abs(gender[i] - y[i])
            if dis < 0.5:
                acc_num = acc_num + 1
        

    print('train_acc:', acc_num/total_num)
    return total_loss / len(loader), acc_num / total_num


def test(model, loader, device, numz):
    model.eval()

    total_loss = 0
    latent_all = []
    label_all = []
    latent_all = np.array(latent_all)
    acc_count = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            pred, gender, pz, z = model(data)
            y = data.y.to(device).float()
            x = data.x.to(device)
            criterion = torch.nn.BCELoss()
            loss = F.l1_loss(pred, x, reduction='mean') + 0.15*criterion(gender, y) + 0.6*F.mse_loss(z, pz)
            total_loss += loss.item()
            if abs(data.y - gender.cpu()) < 0.5:
                acc_count = acc_count + 1
            latent_out = z.detach().cpu().numpy()
            label_y = data.y.float()
            label_y = torch.squeeze(label_y)
            label_out = label_y.detach().cpu().numpy()
            label_all= np.append(label_all, label_out)
            latent_all= np.append(latent_all, latent_out)

    sns.set(context="paper", style="white")
    reducer = umap.UMAP(random_state=42)
    latent_all = np.array(latent_all)
    label_all = np.array(label_all)
    latent_all = latent_all.reshape((np.size(label_all),numz))
    #print(np.size(latent_all))
    #print(np.size(label_all))
    embedding = reducer.fit_transform(latent_all)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1],c=label_all, cmap="Spectral")
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("latent Z data with dimension 64 embedded into 2 dimensions by UMAP", fontsize=18)
    plt.savefig("mygraph.png")  
    plt.close()

    print('test_acc:', acc_count/np.size(label_all))
    return total_loss / len(loader), acc_count/np.size(label_all)


def eval_error(model, test_loader, device, meshdata, out_dir, mesh):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred, _, _, _ = model(data)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_pred_n = pred.cpu().numpy() * std.numpy() + mean.numpy()
            reshaped_x_n = x.cpu().numpy() * std.numpy() + mean.numpy()
            reshaped_pred_n = np.squeeze(reshaped_pred_n)
            reshaped_x_n = np.squeeze(reshaped_x_n)
            #print(np.shape(reshaped_x_n))
            if i % 200 == 0:
                result_mesh = Mesh(v=reshaped_pred_n, f=mesh.f)
                expected_mesh = Mesh(v=reshaped_x_n, f=mesh.f)
                result_mesh.write_ply('resul'+str(i)+'.ply',ascii=True)
                expected_mesh.write_ply('original'+str(i)+'.ply',ascii=True)


            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
