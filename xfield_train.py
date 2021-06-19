# Package Import
import torch
import torch.nn as nn
import numpy as np
import os
from easydict import EasyDict
import cv2
import time

# Module Import
from xfield_model import Net
from load_data import load_data
import flow_vis
from load_video_data import load_video_data

# args = EasyDict({
#     'dataset': './data/t5',
#     'savedir': './results/t5',
#     'type': ['light', 'view', 'time'],
#     'dims': [3, 3, 3],
#     'DSfactor': 8,
#     'neighbor_num': 2,
#     'lr': 0.0001,
#     'sigma': 0.1,
#     'stop_l1_thr': 0.01 
# })

# args = EasyDict({
#     'dataset': './data/t6',
#     'savedir': './results/t6',
#     'type': ['view'],
#     'dims': [3],
#     'DSfactor': 12,
#     'neighbor_num': 2,
#     'lr': 0.0001,
#     'sigma': 0.1,
#     'stop_l1_thr': 0.01,
#     'stop_delta_l1_thr': 0.0005
# })

args = EasyDict({
    'dataset': './data/video/40倍.mp4',
    'savedir': './results/video',
    'video': True,
    'type': ['time'],
    'dims': [3],
    'DSfactor': 4,
    'neighbor_num': 2,
    'lr': 0.0001,
    'sigma': 0.1,
    'stop_l1_thr': 0.01,
    'stop_delta_l1_thr': 0.0005
})



print('------------ prepare data ------------')

dataset = args.dataset

savedir = args.savedir
if not os.path.exists(savedir):
    os.mkdir(savedir)

if not os.path.exists(os.path.join(savedir,'trained model')):
    os.mkdir( os.path.join(savedir,'trained model') )
    print('creating directory %s'%(os.path.join(savedir,'trained model')))

if not os.path.exists(os.path.join(savedir,'saved training')):
    os.mkdir( os.path.join(savedir,'saved training') )
    print('creating directory %s'%(os.path.join(savedir,'saved training')))

print(args.type)

if(args.video):
    img_data, coordinates, training_pairs, img_h, img_w, frames = load_video_data(args)
    args.dims = [frames]
else:
    img_data, coordinates, training_pairs, img_h, img_w = load_data(args)


dims = args.dims
neighbor_num = args.neighbor_num
coord_min = np.min(coordinates)
coord_max = np.max(coordinates)

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, img_data, coordinates, training_pairs, neighbor_num):
        self.coords = torch.from_numpy(coordinates).permute(0,3,1,2).float()
        self.imgs = torch.from_numpy(img_data).permute(0,3,1,2).float()
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

model = Net(img_h, img_w, args)
if(use_gpu):
    model = model.cuda()
    print('<Net> Use GPU')
else:
    print('<Net> Use CPU')

criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0)

model.apply(weight_init)

def train(epoch):
    model.train()

    l1_loss_cumulated = 0

    for i, data in enumerate(train_loader):
        inputs, ref, neighbors, albedo = data
        if use_gpu:
            inputs, ref, neighbors, albedo = inputs.cuda(), ref.cuda(), neighbors.cuda(), albedo.cuda()
        
        inputs = torch.autograd.Variable(inputs)
        ref = torch.autograd.Variable(ref)
        neighbors = torch.autograd.Variable(neighbors)
        albedo = torch.autograd.Variable(albedo)

        optimizer.zero_grad()
        output, flows = model(inputs[0], neighbors[0], albedo)

        loss = criterion(output, ref[0])
        loss.backward()
        optimizer.step()

        l1_loss_cumulated += loss.item()
        if i % 2 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tL1 Loss Cumulated: {:.6f}'.format(
               epoch, (i+1) * len(inputs), len(train_loader.dataset),
               100. * (i+1) / len(train_loader), l1_loss_cumulated), end=' ')
    
    model.eval()
    print('')

    if(args.type == ['light','view','time']):

        center = len(img_data) // 2
        center_img = img_data[center,::]
        cv2.imwrite('{}/saved training/reference.png'.format(savedir, epoch), np.uint8(center_img*255))

        pair = training_pairs[3*center + 0,::]
        output, flows = model(
            torch.from_numpy(coordinates[pair[:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(), 
            torch.from_numpy(img_data[pair[1:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
            pair[-1])

        flow = flows.cpu().detach()[0,0:2,::].permute(1,2,0).numpy()
        flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        cv2.imwrite('{}/saved training/flow_light.png'.format(args.savedir), np.uint8(flow_color))

        flow = flows.cpu().detach()[0,2:4,::].permute(1,2,0).numpy()
        flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        cv2.imwrite('{}/saved training/flow_view.png'.format(args.savedir), np.uint8(flow_color))

        flow = flows.cpu().detach()[0,4:6,::].permute(1,2,0).numpy()
        flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        cv2.imwrite('{}/saved training/flow_time.png'.format(args.savedir), np.uint8(flow_color))
        
        img_out = output.cpu().detach()[0,::].permute(1,2,0).numpy()
        cv2.imwrite('{}/saved training/recons_light_epoch_{}.png'.format(args.savedir, epoch), np.uint8(img_out*255))

        pair = training_pairs[3*center + 1,::]
        output, flows = model(
            torch.from_numpy(coordinates[pair[:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(), 
            torch.from_numpy(img_data[pair[1:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
            pair[-1])
                
        img_out = output.cpu().detach()[0,::].permute(1,2,0).numpy()
        cv2.imwrite('{}/saved training/recons_view_epoch_{}.png'.format(args.savedir, epoch), np.uint8(img_out*255))

        pair = training_pairs[3*center + 2,::]
        output, flows = model(
            torch.from_numpy(coordinates[pair[:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(), 
            torch.from_numpy(img_data[pair[1:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
            pair[-1])
                
        img_out = output.cpu().detach()[0,::].permute(1,2,0).numpy()
        cv2.imwrite('{}/saved training/recons_time_epoch_{}.png'.format(args.savedir, epoch), np.uint8(img_out*255))
    
    else:
        center = len(img_data) // 2
        center_img = img_data[center,::]
        cv2.imwrite('{}/saved training/reference.png'.format(savedir, epoch), np.uint8(center_img*255))

        pair = training_pairs[center,::]
        output, flows = model(
            torch.from_numpy(coordinates[pair[:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(), 
            torch.from_numpy(img_data[pair[1:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
            pair[-1])

        flow = flows.cpu().detach()[0,::].permute(1,2,0).numpy()
        flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        cv2.imwrite('{}/saved training/flow_{}.png'.format(args.savedir, args.type[0]), np.uint8(flow_color))

        img_out = output.cpu().detach()[0,::].permute(1,2,0).numpy()
        cv2.imwrite('{}/saved training/recons_{}_epoch_{}.png'.format(args.savedir, args.type[0], epoch), np.uint8(img_out*255))


    return l1_loss_cumulated / epoch_size

    



iter_end = 100000
epoch_size = len(train_dataset)
epoch_end = iter_end // epoch_size

start_time = time.time()
avg_loss = 1
last_avg_loss = -1
delta_avg_loss = 0
epoch = 0

print('<Train> Total epochs:', epoch_end)

print('\n------------ start training ------------')

while(epoch <= epoch_end and avg_loss >= args.stop_l1_thr):
    avg_loss = train(epoch)
    if(last_avg_loss != -1):
        delta_avg_loss += avg_loss - last_avg_loss
    last_avg_loss = avg_loss

    print('\nElapsed time {:3.1f} m\tAveraged L1 loss = {:3.5f}\tAveraged delta L1 loss = {:3.5f}'.format((time.time()-start_time)/60, avg_loss, delta_avg_loss / (epoch + 1)))
    
    if(delta_avg_loss != 0 and np.abs(delta_avg_loss / epoch) < args.stop_delta_l1_thr):
        break

    epoch += 1
    if(epoch == epoch_end // 2):
        for g in optimizer.param_groups:
            g['lr'] = 0.00005

print('\n------------ save trained model ------------')

trained_model_filename = os.path.join(savedir,'trained model/trained_model.pt')
torch.save(model.state_dict(), trained_model_filename)
