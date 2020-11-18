import time
import os
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
import numpy as np

def eval_reconstruction(model, test_loader, device, meshdata, out_dir, mesh, numz):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    count0, count1 = 0,0
    z0, z1 = [], []
    list_0 = []
    list_1 = []

    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 4 == 0:
                x = data.x.to(device)
                pred, gender, bz,gz, z,_,bmi = model(data)
                imageid1 = int(data.pos[:,0].cpu().numpy())
                #print(imageid1)
                imageid2 = int(data.pos[:,1].cpu().numpy())
                count1 = count1 + 1
                #np.save('../z_test/'+str(imageid1).zfill(6)+str(imageid2).zfill(6), save_z)
                pred = np.squeeze(pred)
                out0 = pred.cpu().numpy() * std.numpy() + mean.numpy()
                result_mesh = Mesh(v=out0, f=mesh.f)
                result_mesh.write_ply('results/rec' +str(i) + '.ply',ascii=True)
                x = np.squeeze(x)
                source_x = x.cpu().numpy() * std.numpy() + mean.numpy()
                result_mesh2 = Mesh(v=source_x, f=mesh.f)
                result_mesh2.write_ply('results/src' + str(i) + '.ply',ascii=True)
                print(i, str(imageid1).zfill(6)+str(imageid2).zfill(6))

                
