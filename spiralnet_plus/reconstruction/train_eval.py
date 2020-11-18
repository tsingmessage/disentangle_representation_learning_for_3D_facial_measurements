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
        out, gender, bz, gz,wz,hz, z ,_,bmi,wei,hei= model(data)
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
            pred, gender,bz,gz ,wz,hz,z, gender_z,bmi,wei,hei= model(data)
            x = data.x.to(device)
            loss = F.l1_loss(pred, x, reduction='mean')
            total_loss += loss.item()
 
    return total_loss / len(loader)


def eval_error(model, test_loader, device, meshdata, out_dir, mesh):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred, _, _, _,_,_ ,_= model(data)
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
