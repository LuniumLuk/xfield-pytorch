from numpy.core.fromnumeric import size, std
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import os
from easydict import EasyDict
from torch.nn.modules.upsampling import Upsample
from load_data import load_data
import cv2
import flow_vis

import time

args = EasyDict({
    'dataset': './data',
    'savedir': './results',
    'type': ['light', 'time', 'view'],
    'dims': [3, 3, 3],
    'DSfactor': 12,
    'neighbor_num': 2,
    'lr': 0.0001,
    'sigma': 0.1
})

print('------------ prepare data ------------')

dataset = args.dataset

savedir = args.savedir
if not os.path.exists(savedir):
    os.mkdir(savedir)

if not os.path.exists(os.path.join(savedir,"trained model") ):
    os.mkdir( os.path.join(savedir,"trained model") )
    print('creating directory %s'%(os.path.join(savedir,"trained model")))

if not os.path.exists(os.path.join(savedir,"saved training") ):
    os.mkdir( os.path.join(savedir,"saved training") )
    print('creating directory %s'%(os.path.join(savedir,"saved training")))

print(args.type)

img_data, coordinates, training_pairs, img_h, img_w = load_data(args)



dims = args.dims
neighbor_num = args.neighbor_num
coord_min = np.min(coordinates)
coord_max = np.max(coordinates)

print('\n------------ establish model ------------')

class Net(torch.nn.Module):
    def __init__(self, img_h, img_w):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w

        pad_x, pad_y, up_x, up_y = self.generate_factors(self.img_h, self.img_w)
        ngf = 4
        layer_channels = [ngf*16, ngf*16, ngf*16, ngf*8, ngf*8, ngf*8, ngf*4]
        self.layer_num = len(pad_x)
        layer_channels.extend([ngf*4] * (self.layer_num - len(layer_channels))) 

        in_channels = [3, ngf*16+2]
        in_channels.extend(layer_channels[1:-1])

        net_list = []

        for i in range(self.layer_num):
            if(i == 0):
                net_list.append(nn.Upsample(scale_factor=(up_y[i], up_x[i])))
                net_list.append(nn.ReflectionPad2d(padding=(0,pad_x[i],0,pad_y[i])))
                net_list.append(nn.Conv2d(in_channels=in_channels[i], out_channels=layer_channels[i], kernel_size=1, stride=1))
                net_list.append(nn.LeakyReLU())
            else:
                net_list.append(nn.Upsample(scale_factor=(up_y[i], up_x[i])))
                net_list.append(nn.ReflectionPad2d(padding=(1,1+pad_x[i],1,1+pad_y[i])))
                net_list.append(nn.Conv2d(in_channels=in_channels[i], out_channels=layer_channels[i], kernel_size=3, stride=1))
                net_list.append(nn.LeakyReLU())
        
        self.nets = nn.ModuleList(net_list)
            

        self.coordconv = torch.tensor(
            [[[[0, 2],
               [0, 2]],
              [[0, 0],
               [2, 2]]]], dtype=torch.float32).cuda()
        self.coord_pad = nn.ReflectionPad2d(padding=(0,pad_x[0],0,pad_y[0]))

        self.flow_pad = nn.ReflectionPad2d((1,1,1,1))
        self.flow_conv2d = nn.Conv2d(in_channels=ngf*4, out_channels=6, kernel_size=3, stride=1)
        self.flow_activate = nn.Tanh()

        flow_nparams = self.get_parameter_num()
        print('<Net> decoder params count:', flow_nparams)

        albedos_data = np.ones((9, 3, self.img_h, self.img_w))
        self.albedos = (torch.nn.Parameter(data=torch.from_numpy(albedos_data), requires_grad=True))

        albedo_nparams = self.get_parameter_num() - flow_nparams
        print('<Net> albedos params count:', albedo_nparams)

    def blending(self, x, neighbors, flows, albedo):

        epsilon = 0.00001
        
        img_h = self.img_h
        img_w = self.img_w

        light_flow = flows[:1,0:2,:,:]
        view_flow = flows[:1,2:4,:,:]
        time_flow = flows[:1,4:6,:,:]

        light_flow_neighbor = flows[1:,0:2,:,:]
        view_flow_neighbor = flows[1:,2:4,:,:]
        time_flow_neighbor = flows[1:,4:6,:,:]

        coord_input = x[:1,::]
        coord_neighbor = x[1:,::]


        delta = torch.tile(coord_input - coord_neighbor, (1, 1, img_h, img_w))
        delta_light = delta[:,0:1,:,:]
        delta_view = delta[:,1:2,:,:]
        delta_time = delta[:,2:3,:,:]

        flag = (torch.abs(delta_light) > 0).float()
        offset_forward = delta_light*light_flow + delta_view*view_flow + delta_time*time_flow
        # where there's shift in light, use neighbors's albedo
        # otherwise, use trained albedo * neighbors' shading
        # print('flag.shape', flag.shape)
        # print('neighbors.shape', neighbors.shape)
        # print('albedo.shape', albedo.shape)
        shading = flag*neighbors/albedo + (1-flag)*neighbors 

        warped_shading = torch.nn.functional.grid_sample(shading.float(), offset_forward.permute(0,2,3,1), mode='bilinear', padding_mode='border', align_corners=False)
        warped_view_flow = torch.nn.functional.grid_sample(view_flow_neighbor, offset_forward.permute(0,2,3,1), mode='bilinear', padding_mode='border', align_corners=False)
        warped_time_flow = torch.nn.functional.grid_sample(time_flow_neighbor, offset_forward.permute(0,2,3,1), mode='bilinear', padding_mode='border', align_corners=False)
        warped_light_flow = torch.nn.functional.grid_sample(light_flow_neighbor, offset_forward.permute(0,2,3,1), mode='bilinear', padding_mode='border', align_corners=False)

        warped_image = flag*warped_shading*albedo + (1-flag)*warped_shading

        offset_backward = delta_light*warped_light_flow + delta_view*warped_view_flow + delta_time*warped_time_flow

        dist = torch.sum(torch.abs(offset_forward-offset_backward), dim=1, keepdim=True)
        weight = torch.exp(-args.sigma*img_w*dist)
        weight_normalized = weight/(torch.sum(weight,dim=0,keepdim=True) + epsilon)
        interpolated = torch.sum(torch.multiply(warped_image, weight_normalized), dim=0, keepdim=True)

        return interpolated


    def generate_factors(self, h, w):
        temp = h
        pad_y = [temp % 2]
        while(temp != 1):
            temp //= 2
            pad_y.append(temp % 2)
        del pad_y[-1]
        pad_y.reverse()

        temp = w
        pad_x = [temp % 2]
        while(temp != 1):
            temp //= 2
            pad_x.append(temp % 2)
        del pad_x[-1]
        pad_x.reverse()

        len_x = len(pad_x)
        len_y = len(pad_y)

        up_x = [2] * len_x
        up_y = [2] * len_y

        if(len_x > len_y):
            pad_y.extend([0] * (len_x - len_y))
            up_y.extend([1] * (len_x - len_y))
        
        if(len_y > len_x):
            pad_x.extend([0] * (len_y - len_x))
            up_x.extend([1] * (len_y - len_x))
        
        return pad_x, pad_y, up_x, up_y

    def get_parameter_num(self):
        total_param_count = 0
        for param in self.parameters():
            count = 1
            for s in param.size():
                count *= s
            total_param_count += count
        return total_param_count

    def forward(self, x, neighbors, albedo_index):

        input_x = torch.clone(x)

        # print('input', x.shape)

        for i in range(self.layer_num):
            # print('--- layer %d ---' % i)
            x = self.nets[i * 4](x)
            # print(x.shape)
            x = self.nets[i * 4 + 1](x)
            # print(x.shape)
            x = self.nets[i * 4 + 2](x)
            # print(x.shape)
            x = self.nets[i * 4 + 3](x)
            # print(x.shape)
            if(i == 0):
                coordconv_tl = torch.tile(self.coordconv, [3,1,1,1])
                coordconv_tl = self.coord_pad(coordconv_tl)
                # print(x.shape, coordconv_tl.shape)
                x = torch.cat((x, coordconv_tl), dim=1)
                # print(x.shape)
        
        x = self.flow_pad(x)
        # print(x.shape)
        x = self.flow_conv2d(x)
        # print(x.shape)
        x = self.flow_activate(x)
        # print(x.shape)

        albedo_index = albedo_index.int().item()
        albedo = self.albedos[albedo_index,::]
        # print('albedo shape', albedo.shape)
        interpolated = self.blending(input_x, neighbors, x, albedo)
        
        return interpolated, x

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, img_data, coordinates, training_pairs, neighbor_num):
        self.coords = torch.from_numpy(coordinates).permute(0,3,1,2)
        self.imgs = torch.from_numpy(img_data).permute(0,3,1,2)
        self.pairs = training_pairs
        self.n_num = neighbor_num

        if(len(self.pairs) < 500):
            self.pairs = np.repeat(self.pairs, 500//len(self.pairs), axis=0)
        self.len = self.pairs.shape[0]
    
    def __getitem__(self, index):
        pair = self.pairs[index,::]
        return self.coords[pair[:self.n_num+1],::], self.imgs[pair[:1],::], self.imgs[pair[1:self.n_num+1],::], pair[-1]

    def __len__(self):
        return self.len


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)


train_dataset = TrainDataset(img_data, coordinates, training_pairs, neighbor_num)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)


use_gpu = torch.cuda.is_available()

model = Net(img_h, img_w)
if(use_gpu):
    model = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')

criterion = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0)

model.apply(weight_init)



def train(epoch):
    model.train()

    l1_loss_cumulated = 0

    for i, data in enumerate(train_loader):
        inputs, ref, neighbors, albedo = data
        if use_gpu:
            inputs, ref, neighbors, albedo = inputs.cuda(), ref.cuda(), neighbors.cuda(), albedo.cuda()
        
        inputs = torch.autograd.Variable(inputs).float()
        ref = torch.autograd.Variable(ref)
        neighbors = torch.autograd.Variable(neighbors).float()
        albedo = torch.autograd.Variable(albedo)

        optimizer.zero_grad()
        output, flows = model(inputs[0], neighbors[0], albedo)

        loss = criterion(output, ref[0])
        loss.backward()
        optimizer.step()

        l1_loss_cumulated += loss.item()
        if i % 2 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tL1 Loss Cumulated: {:.6f}'.format(
               epoch, i * len(inputs), len(train_loader.dataset),
               100. * i / len(train_loader), l1_loss_cumulated), end=' ')
    
    flow = flows.cpu().detach()[0,0:2,::].permute(1,2,0).numpy()
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    cv2.imwrite('{}/saved training/flow_light_epoch{}.png'.format(args.savedir, epoch), np.uint8(flow_color))

    flow = flows.cpu().detach()[0,2:4,::].permute(1,2,0).numpy()
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    cv2.imwrite('{}/saved training/flow_view_epoch{}.png'.format(args.savedir, epoch), np.uint8(flow_color))

    flow = flows.cpu().detach()[0,4:6,::].permute(1,2,0).numpy()
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    cv2.imwrite('{}/saved training/flow_time_epoch{}.png'.format(args.savedir, epoch), np.uint8(flow_color))
    
    print('\nElapsed time {:3.1f} m\tAveraged L1 loss = {:3.5f}'.format((time.time()-start_time)/60, l1_loss_cumulated / epoch_size))
    img_out = output.cpu().detach()[0,::].permute(1,2,0).numpy() * 255
    filename = '{}/saved training/recon_epoch{}.png'.format(args.savedir, epoch)
    print('Image saved: ', filename)
    cv2.imwrite(filename, np.uint8(img_out))
            



iter_end = 100000
stop_l1_thr = 0.01
epoch_size = len(train_dataset)
epoch_end = iter_end // epoch_size

start_time = time.time()
min_loss = 1000
l1_loss_t = 1
epoch = 0

# def test_train():
#     model.train()		

#     for i, data in enumerate(train_loader):
#         inputs, ref, neighbors, albedo = data
#         if use_gpu:
#             inputs, ref, neighbors, albedo = inputs.cuda(), ref.cuda(), neighbors.cuda(), albedo.cuda()
        
#         inputs = torch.autograd.Variable(inputs).float()
#         ref = torch.autograd.Variable(ref)
#         neighbors = torch.autograd.Variable(neighbors).float()
#         albedo = torch.autograd.Variable(albedo)

#         print(inputs.shape, ref.shape, neighbors.shape, albedo.shape)


# test_train()

print('\n------------ start training ------------')

while(epoch <= epoch_end):
    train(epoch)
    epoch += 1
    if(epoch == epoch_end//2):
        args.lr = 0.00005
