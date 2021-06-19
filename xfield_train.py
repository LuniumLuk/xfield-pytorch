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

args = EasyDict({
    'dataset': './data/3x3x3/mydata',
    'savedir': './results/3x3x3/mydata',
    'type': ['light', 'time', 'view'],
    'dims': [3, 3, 3],
    'DSfactor': 16,      # 降采样倍数，降采样倍数越大，图片会变小，训练速度会快很多
    'neighbor_num': 2,  # 训练时侯用多少个邻居做插值，time =1 那么 time = 0 / 2 做插值
    'lr': 0.0001,       # learning rate 步长
    'sigma': 0.1,       #
    'stop_l1_thr': 0.01  # 训练loss损失函数到0.01的时候停止训练
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

img_data, coordinates, training_pairs, img_h, img_w = load_data(args)

# n 一个batch数据数量 c 通道 h 高 w 宽 （四维张量）
# img_data shape 27图片数目 * 512 * 512 * 3通道
# coordinates 坐标 27 * 1 * 1 * 3
# training_pairs 对于每一张图片，在每一个维度上它的邻居和自己 81 * 3 * 3
# img_h,img_w 训练时候的大小

dims = args.dims
neighbor_num = args.neighbor_num
coord_min = np.min(coordinates)   # 0
coord_max = np.max(coordinates)   # 2

## 准备数据
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, img_data, coordinates, training_pairs, neighbor_num):
        self.coords = torch.from_numpy(coordinates).permute(0,3,1,2).float()   # 调整维度顺序，把他变成张量
        self.imgs = torch.from_numpy(img_data).permute(0,3,1,2).float()
        self.pairs = training_pairs
        self.n_num = neighbor_num

        if(len(self.pairs) < 500):
            self.pairs = np.repeat(self.pairs, 500//len(self.pairs), axis=0)   # 复制500个左右，扩充训练
        self.len = self.pairs.shape[0]

    def __getitem__(self, index):
        # 每次训练只训练一张图片
        pair = self.pairs[index,::]
        return self.coords[pair[:self.n_num+1],::], self.imgs[pair[:1],::], self.imgs[pair[1:self.n_num+1],::], pair[-1]
                    # 所有参与训练图片的坐标（一组坐标）【本身图片，邻居1，邻居2】
                    # 自己的这张图片
                    # 邻居图片
                    # 颜色序号的坐标是第几维
    def __len__(self):
        return self.len

# 初始化所有权重，用的是标准分布
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
        inputs, ref, neighbors, albedo = data  # get_item 返回的一组
        if use_gpu:
            inputs, ref, neighbors, albedo = inputs.cuda(), ref.cuda(), neighbors.cuda(), albedo.cuda()

        inputs = torch.autograd.Variable(inputs)      # 标准流程
        ref = torch.autograd.Variable(ref)
        neighbors = torch.autograd.Variable(neighbors)
        albedo = torch.autograd.Variable(albedo)

        optimizer.zero_grad()
        output, flows = model(inputs[0], neighbors[0], albedo)
        # output 是用邻居 预测 本身的图片得到的结果，flows图是流图

        loss = criterion(output, ref[0])
        loss.backward()
        optimizer.step()

        l1_loss_cumulated += loss.item()
        if i % 2 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tL1 Loss Cumulated: {:.6f}'.format(
               epoch, (i+1) * len(inputs), len(train_loader.dataset),
               100. * (i+1) / len(train_loader), l1_loss_cumulated), end=' ')

    print('\nElapsed time {:3.1f} m\tAveraged L1 loss = {:3.5f}'.format((time.time()-start_time)/60, l1_loss_cumulated / epoch_size))

    model.eval() # 切换到输出模式

    # 用来输出 取了所有维度中 最中间的图片

    center = len(img_data) // 2
    center_img = img_data[center,::]
    cv2.imwrite('{}/saved training/reference.png'.format(savedir, epoch), np.uint8(center_img*255))

    pair = training_pairs[3*center + 0,::]
    output, flows = model(
        torch.from_numpy(coordinates[pair[:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
        torch.from_numpy(img_data[pair[1:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
        pair[-1])
    # output是插值好的照片，flows是流的图片， 维度为 1 * 6(前两个是light维度的flow，中间两个是view维度的flow，后面两个是time维度的flow) * h * w

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
        torch.from_numpy(coordinates[pair[:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),   # 变回来，方便cv2输出
        torch.from_numpy(img_data[pair[1:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
        pair[-1])

    img_out = output.cpu().detach()[0,::].permute(1,2,0).numpy()   # detach() 从训练的东西中拿出图片

    cv2.imwrite('{}/saved training/recons_view_epoch_{}.png'.format(args.savedir, epoch), np.uint8(img_out*255))

    pair = training_pairs[3*center + 2,::]
    output, flows = model(
        torch.from_numpy(coordinates[pair[:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
        torch.from_numpy(img_data[pair[1:args.neighbor_num+1],::]).permute(0,3,1,2).float().cuda(),
        pair[-1])

    img_out = output.cpu().detach()[0,::].permute(1,2,0).numpy()
    cv2.imwrite('{}/saved training/recons_time_epoch_{}.png'.format(args.savedir, epoch), np.uint8(img_out*255))

    return l1_loss_cumulated / epoch_size





iter_end = 100000
epoch_size = len(train_dataset)
epoch_end = iter_end // epoch_size

start_time = time.time()
avg_loss = 1
epoch = 0

print('<Train> Total epochs:', epoch_end)

print('\n------------ start training ------------')

while(epoch <= epoch_end and avg_loss >= args.stop_l1_thr):
    avg_loss = train(epoch)
    epoch += 1
    if(epoch == epoch_end // 2):
        for g in optimizer.param_groups:
            g['lr'] = 0.00005

print('\n------------ save trained model ------------')

trained_model_filename = os.path.join(savedir,'trained model/trained_model.pt')
torch.save(model.state_dict(), trained_model_filename)
