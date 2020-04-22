import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from spiralconv import SpiralConv


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out


class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, numz, device):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        # latent generator
        self.latent_generator_g = torch.nn.Linear(1, int(numz/4))
        self.latent_generator_b = torch.nn.Linear(1, int(numz/4))
        self.latent_generator_w = torch.nn.Linear(1, int(numz/4))
        self.latent_generator_h = torch.nn.Linear(1, int(numz/4))
        # encoder
        self.en_layers = nn.ModuleList()
        self.device = device
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # regressor b
        self.re_layers_b = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.re_layers_b.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.re_layers_b.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.re_layers_b.append(
            nn.Linear(self.num_vert * out_channels[-1], 1))
        # regressor w
        self.re_layers_w = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.re_layers_w.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.re_layers_w.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.re_layers_w.append(
            nn.Linear(self.num_vert * out_channels[-1], 1))
        # regressor h
        self.re_layers_h = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.re_layers_h.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.re_layers_h.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.re_layers_h.append(
            nn.Linear(self.num_vert * out_channels[-1], 1))
        # classification
        self.cl_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.cl_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.cl_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.cl_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], 1))
        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
                if i == 3 :
                    gender = x.clone()   
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        my_drop = torch.nn.Dropout(p=0.5)
        #gender = my_drop(gender)
        #gender = self.re_layers_b[3](gender, self.down_transform[3]) #additional convolution layer for regressor
        gender_z = gender.view(-1, layer.weight.size(1))
        
        gender_z = my_drop(gender_z)
        bmi = self.re_layers_b[4](gender_z) #output layer for regressor
        wei = self.re_layers_w[4](gender_z) #output layer for regressor
        hei = self.re_layers_h[4](gender_z) #output layer for regressor        
        gender = self.cl_layers[4](gender_z) #output layer for regressor
        gender = torch.sigmoid(gender)
        return x, gender, gender_z,bmi,wei,hei

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def forward(self, data, *indices):
        x = data.x.to(self.device)
        gender_t = data.y.to(self.device).float()
        bmi_t = data.pos[:,0].to(self.device).float()
        weight_t = data.pos[:,1].to(self.device).float()
        height_t = data.pos[:,2].to(self.device).float()
        len_batch,_ = gender_t.size()
        bmi_t = torch.reshape(bmi_t,(len_batch,1))
        weight_t = torch.reshape(weight_t,(len_batch,1))
        height_t = torch.reshape(height_t,(len_batch,1))
        z, gender,gender_z,bmi,wei,hei = self.encoder(x)
        out = self.decoder(z)
        bz = self.latent_generator_b(bmi_t)
        gz = self.latent_generator_g(gender_t)
        wz = self.latent_generator_w(weight_t)
        hz = self.latent_generator_h(height_t)
        return out, gender , bz,gz,wz,hz, z,gender_z,bmi,wei,hei
