import time
import os
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
import numpy as np
import seaborn as sns
import umap
import matplotlib.pyplot as plt


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
            loss = F.l1_loss(pred, x, reduction='mean') #+ 0.2*criterion(gender, y) + 0.4*F.mse_loss(z, pz)
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
    print(np.size(latent_all))
    print(np.size(label_all))
    embedding = reducer.fit_transform(latent_all)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1],c=label_all, cmap="Spectral")
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("latent Z data with dimension 64 embedded into 2 dimensions by UMAP", fontsize=18)
    plt.savefig("mygraph.png")  
    plt.close()

    print(acc_count/np.size(label_all))
    return total_loss / len(loader), acc_count/np.size(label_all)


def eval_error(model, test_loader, device, meshdata, out_dir, mesh, numz):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    z0, z1 = [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            y = data.y
            pred, gender, pz, z = model(data)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_pred_n = pred.cpu().numpy() * std.numpy() + mean.numpy()
            reshaped_x_n = x.cpu().numpy() * std.numpy() + mean.numpy()
            reshaped_pred_n = np.squeeze(reshaped_pred_n)
            reshaped_x_n = np.squeeze(reshaped_x_n)
            if abs(y - gender.cpu() )>  0.1:
                continue
            if y == 1:
                p1 = z
                z1.append(z.unsqueeze(0))
            else:
                p0 = z
                z0.append(z.unsqueeze(0))







            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]

            errors.append(tmp_error)
#-------------------------------------
        z0_tensor = torch.cat(z0)
        z1_tensor = torch.cat(z1)
        num0, _ ,_ = z0_tensor.size()
        num1, _ ,_ = z1_tensor.size()
        z0_tensor =z0_tensor.view(num0,numz)
        z1_tensor =z1_tensor.view(num1,numz)
        print(z0_tensor.size())
        z0_mean = torch.mean(z0_tensor,dim=0)
        z0_max ,_= torch.max(z0_tensor,dim=0)
        z0_min ,_= torch.min(z0_tensor,dim=0)
        print(z0_mean)
        print(z0_max)
        print(z0_min)
        z1_mean = torch.mean(z1_tensor,dim=0)
        z1_max ,_= torch.max(z1_tensor,dim=0)
        z1_min ,_= torch.min(z1_tensor,dim=0)
        z0_mean = z0_mean.view(1,numz)
        z1_mean = z1_mean.view(1,numz)
        print(z1_mean)
        print(z1_max)
        print(z1_min)
        out0 = model.decoder(z0_mean)
        out1  = model.decoder(z1_mean)
        outm  = model.decoder((z0_mean + z1_mean)*0.5)

        out0 = out0.cpu().numpy() * std.numpy() + mean.numpy()
        out0 = np.squeeze(out0)
        result_mesh = Mesh(v=out0, f=mesh.f)
        result_mesh.write_ply('mean0.ply',ascii=True)

        out1 = out1.cpu().numpy() * std.numpy() + mean.numpy()
        out1 = np.squeeze(out1)
        result_mesh = Mesh(v=out1, f=mesh.f)
        result_mesh.write_ply('mean1.ply',ascii=True)

        outm = outm.cpu().numpy() * std.numpy() + mean.numpy()
        outm = np.squeeze(outm)
        result_mesh = Mesh(v=outm, f=mesh.f)
        result_mesh.write_ply('meanm.ply',ascii=True)
        for z_dimension in range(8,64):
        #z_dimension = 7
            diff = z0_max[z_dimension]*1 - z0_min[z_dimension]*1
            for i in range(10):
            #middle = (z0_mean*(i+1)/10  + z1_mean*(1-(i+1)/10) )*1
                step = diff*(i+1)/10    
                z0_mean[0,z_dimension] = z0_min[z_dimension]*1 + step
            #print(z0_mean)
                out0 = model.decoder(z0_mean)
                out0 = np.squeeze(out0)
                out0 = out0.cpu().numpy() * std.numpy() + mean.numpy()
                result_mesh = Mesh(v=out0, f=mesh.f)
                result_mesh.write_ply('../diff/diff' + str(z_dimension) + '/diff' + str(i) + '.ply',ascii=True)

#-----------------------------------
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
