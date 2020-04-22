import time
import os
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
import numpy as np
import seaborn as sns
import umap
import matplotlib.pyplot as plt

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
    labelg_all = []
    labelb_all = []
    labelw_all = []
    labelh_all = []
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
            #if int(gender_t) == 0:
            if 1:
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

            label_g = gender_t
            label_b = bmi_t
            label_w = wei_t
            label_h = hei_t

            label_g = torch.squeeze(label_g)
            label_b = torch.squeeze(label_b)
            label_w = torch.squeeze(label_w)
            label_h = torch.squeeze(label_h)

            labelg_out = label_g.detach().cpu().numpy()
            labelb_out = label_b.detach().cpu().numpy()
            labelw_out = label_w.detach().cpu().numpy()
            labelh_out = label_h.detach().cpu().numpy()


            labelg_all= np.append(labelg_all, labelg_out)
            labelb_all= np.append(labelb_all, labelb_out)
            labelw_all= np.append(labelw_all, labelw_out)
            labelh_all= np.append(labelh_all, labelh_out)

            latent_all= np.append(latent_all, latent_out)

    sns.set(context="paper", style="white")
    reducer = umap.UMAP(random_state=42)
    #latent_all = np.array(latent_all)
    
    
    latent_all = latent_all.reshape((np.size(labelg_all),numz))
    latent_b = latent_all[:,0:int(numz*0.25)]
    latent_w = latent_all[:,int(numz*0.25):int(numz*0.50)]
    latent_h = latent_all[:,int(numz*0.50):int(numz*0.75)]
    latent_g = latent_all[:,int(numz*0.75):numz]
    #print(np.size(latent_all))
    #print(np.size(label_all))


    embedding = reducer.fit_transform(latent_g)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1],c=labelg_all, cmap="Spectral")
    plt.savefig("gender.png") 
    plt.close()

    embedding = reducer.fit_transform(latent_b)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = mscatter(embedding[:, 0], embedding[:, 1], m=marker_list, ax=ax, c=10*labelb_all,cmap="Spectral")
    plt.savefig("bmi.png") 
    plt.close()

    embedding = reducer.fit_transform(latent_w)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = mscatter(embedding[:, 0], embedding[:, 1], m=marker_list, ax=ax, c=10*labelw_all,cmap="Spectral")
    plt.savefig("wei.png") 
    plt.close()
    embedding = reducer.fit_transform(latent_h)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = mscatter(embedding[:, 0], embedding[:, 1], m=marker_list, ax=ax, c=10*labelh_all,cmap="Spectral")
    plt.savefig("hei.png") 
    plt.close()
    

    print('test_acc b:', acc_count/np.size(labelg_all))
    print('test_acc g:', acc_g/np.size(labelg_all))
    #return total_loss / len(loader), acc_count/np.size(label_all)
    return total_loss / len(loader), total_loss_r / len(loader)


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
            pred, gender, bz,gz, z,_,bmi = model(data)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_pred_n = pred.cpu().numpy() * std.numpy() + mean.numpy()
            reshaped_x_n = x.cpu().numpy() * std.numpy() + mean.numpy()
            reshaped_pred_n = np.squeeze(reshaped_pred_n)
            reshaped_x_n = np.squeeze(reshaped_x_n)
            if bmi.cpu() < 0.3 and gender.cpu() > 0.90:
                p1 = z
                z1.append(z.unsqueeze(0))
            elif bmi.cpu() < 0.3 and gender.cpu() < 0.10:
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
        print(z1_tensor.size())
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



        for z_dimension in range(1):
        #z_dimension = 7
            for i in range(10):
                middle = (z0_mean*(i+1)/10  + z1_mean*(1-(i+1)/10) )*1
                #middle[:,0:32] = z0_mean[:,0:32]
                out0 = model.decoder(middle)
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
