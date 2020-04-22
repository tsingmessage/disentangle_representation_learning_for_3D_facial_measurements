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
        gender_t = data.y.to(device).float()
        bmi_t = data.pos[:,0].to(device).float()
        wei_t = data.pos[:,1].to(device).float()
        hei_t = data.pos[:,2].to(device).float()
        x = data.x.to(device)
        out, gender, bz, gz,wz,hz, z ,_,bmi,wei,hei= model(data)
        criterion = torch.nn.BCELoss()
        bmi = torch.squeeze(bmi)
        wei = torch.squeeze(wei)
        hei = torch.squeeze(hei)
        #print(z.size())
        #halfz=z[:,0:32]
        #print(halfz.size())
        #print(bz.size())
        #exit(0)
        loss = F.l1_loss(out, x, reduction='mean') + 0.35*F.mse_loss(gender, gender_t) +  1*F.mse_loss(z[:,int(numz*0.75):numz], gz)+ 0.35*F.mse_loss(bmi_t, bmi)+ 1*F.mse_loss(z[:,0:int(numz*0.25)], bz) + 0.35*F.mse_loss(wei_t, wei)+ 1*F.mse_loss(z[:,int(numz*0.25):int(numz*0.50)], wz)+ 0.35*F.mse_loss(hei_t, hei)+ 1*F.mse_loss(z[:,int(numz*0.50):int(numz*0.75)], hz) 
        loss.backward()
        loss_rec = F.mse_loss(gender, gender_t)
        total_loss_r += loss_rec.item()
        total_loss += loss.item()
        optimizer.step()

        for i in range(len(bmi)):
            total_num = total_num + 1
            dis = abs(bmi[i] - bmi_t[i])
            if dis < 0.1:
                acc_num = acc_num + 1
            dis = abs(gender[i] - gender_t[i])
            if dis < 0.5:
                acc_g = acc_g + 1
        

    print('train_acc b:', acc_num/total_num)
    print('train_acc g:', acc_g/total_num)
    #return total_loss / len(loader), acc_num / total_num

    return total_loss / len(loader), total_loss_r / len(loader)



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

    total_loss,total_loss_r = 0,0
    latent_all = []
    latent_all2 = []
    label_all = []
    label_all2 = []
    gender_all = []
    latent_all = np.array(latent_all)
    marker_list = []
    acc_count = 0
    acc_g = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            pred, gender,bz,gz ,wz,hz,z, gender_z,bmi,wei,hei= model(data)
            gender_t = data.y.to(device).float()
            bmi_t = data.pos[:,0].to(device).float()
            wei_t = data.pos[:,1].to(device).float()
            hei_t = data.pos[:,2].to(device).float()
            #len_batch,_ = gender_t.size()
            bmi_t = torch.reshape(bmi_t,(1,1))
            wei_t = torch.reshape(wei_t,(1,1))
            hei_t = torch.reshape(hei_t,(1,1))
            #print(bmi_t.size())
            #print(bmi.size())
            #print(data.y)
            if int(gender_t) == 0:
                marker_list = np.append(marker_list, "o")
            else:
                marker_list = np.append(marker_list, '+')
            x = data.x.to(device)
            criterion = torch.nn.BCELoss()
            loss = F.l1_loss(pred, x, reduction='mean') + 0.35*F.mse_loss(gender, gender_t) +  1*F.mse_loss(z[:,int(numz*0.75):numz], gz)+ 0.35*F.mse_loss(bmi_t, bmi)+ 1*F.mse_loss(z[:,0:int(numz*0.25)], bz) + 0.35*F.mse_loss(wei_t, wei)+ 1*F.mse_loss(z[:,int(numz*0.25):int(numz*0.50)], wz)+ 0.35*F.mse_loss(hei_t, hei)+ 1*F.mse_loss(z[:,int(numz*0.50):int(numz*0.75)], hz) 
            loss_rec = F.mse_loss(gender, gender_t)
            total_loss_r += loss_rec.item()
            total_loss += loss.item()
            if abs(bmi_t - bmi) < 0.1:
                acc_count = acc_count + 1
            if abs(gender_t - gender) < 0.5:
                acc_g = acc_g + 1
            latent_out = z.detach().cpu().numpy()
            label_y = bmi_t
            label_y2 = gender_t
            label_y = torch.squeeze(label_y)
            label_y2 = torch.squeeze(label_y2)
            gender_out = torch.squeeze(bmi)
            gender_out = gender_out.detach().cpu().numpy()
            label_out = label_y.detach().cpu().numpy()
            label_out2 = label_y2.detach().cpu().numpy()
            gender_all= np.append(gender_all, gender_out)
            label_all= np.append(label_all, label_out)
            label_all2= np.append(label_all2, label_out2)
            latent_all= np.append(latent_all, latent_out)

    sns.set(context="paper", style="white")
    reducer = umap.UMAP(random_state=42)
    latent_all = np.array(latent_all)
    
    label_all = np.array(label_all)
    latent_all = latent_all.reshape((np.size(label_all),numz))
    latent_b = latent_all[:,0:int(numz*0.25)]
    latent_g = latent_all[:,int(numz*0.75):numz]
    #print(np.size(latent_all))
    #print(np.size(label_all))
    embedding = reducer.fit_transform(latent_b)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = mscatter(embedding[:, 0], embedding[:, 1], m=marker_list, ax=ax, c=10*label_all,cmap="Spectral")
    #plt.scatter(embedding[:, 0], embedding[:, 1],c=10*label_all, cmap="Spectral")
    #plt.setp(ax, xticks=[], yticks=[])
    #plt.title("latent Z data with dimension 64 embedded into 2 dimensions by UMAP", fontsize=18)
     

    #cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    #cbar.set_ticks(np.arange(10))
    #cbar.set_ticklabels(classes)
    plt.savefig("bmi.png") 
    plt.close()

    embedding = reducer.fit_transform(latent_g)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1],c=1*label_all2, cmap="Spectral")
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("latent Z data with dimension 64 embedded into 2 dimensions by UMAP", fontsize=18)
    plt.savefig("gender.png") 
    plt.close()


    
    index_sort = np.argsort(label_all)
    sorted_label = np.sort(label_all)
    sorted_gender = gender_all[index_sort]

    n = np.arange(1,i+2)
    plt.figure()
    plt.plot(sorted_label-sorted_gender)
    plt.ylim(-0.6, 0.6)  
    plt.savefig("regression.png")
    plt.close()
    print('test_acc b:', acc_count/np.size(label_all))
    print('test_acc g:', acc_g/np.size(label_all))
    #return total_loss / len(loader), acc_count/np.size(label_all)
    return total_loss / len(loader), total_loss_r / len(loader)


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
