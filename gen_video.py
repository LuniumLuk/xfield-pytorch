import torch
import os
import numpy as np
from xfield_model import Net
from load_data import load_data
from easydict import EasyDict
import cv2

args = EasyDict({
    'dataset': './data/t2',
    'savedir': './results/t2',
    'type': ['light', 'time', 'view'],
    'dims': [3, 3, 3],
    'DSfactor': 12,
    'neighbor_num': 2,
    'lr': 0.0001,
    'sigma': 0.1,
    'stop_l1_thr': 0.01 
})

img_data, coordinates, training_pairs, img_h, img_w = load_data(args)

use_gpu = torch.cuda.is_available()

model = Net(img_h, img_w, args)
if(use_gpu):
    model = model.cuda()
    print('<Net> Use GPU')
else:
    print('<Net> Use CPU')

model.load_state_dict(torch.load(os.path.join(args.savedir,'trained model/trained_model')))
model.eval()

def gen_video():

    precomputed_flows = []

    for i in range(len(coordinates)):
        flows_out = model(torch.from_numpy(coordinates[[i],::]).permute(0,3,1,2).float().cuda(), flow=True)
        precomputed_flows.append(flows_out[0,::].cpu().detach().numpy())
      
    precomputed_flows = np.stack(precomputed_flows,0)

    print(precomputed_flows.shape)


gen_video()