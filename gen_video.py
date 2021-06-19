import torch
import os
import numpy as np
from xfield_model import Net
from load_data import load_data
from easydict import EasyDict
import cv2

# args = EasyDict({
#     'dataset': './data/t5',
#     'savedir': './results/t5',
#     'type': ['light', 'time', 'view'],
#     'dims': [3, 3, 3],
#     'DSfactor': 8,
#     'neighbor_num': 2,
#     'lr': 0.0001,
#     'sigma': 0.1,
#     'stop_l1_thr': 0.01 
# })

args = EasyDict({
    'dataset': './data/t6',
    'savedir': './results/t6',
    'type': ['view'],
    'dims': [3],
    'DSfactor': 12,
    'neighbor_num': 2,
    'lr': 0.0001,
    'sigma': 0.1,
    'stop_l1_thr': 0.01 
})

num_n = 2
dims = args.dims
scale = 90
fps = 60

if not os.path.exists(os.path.join(args.savedir,'rendered videos')):
     os.mkdir(os.path.join(args.savedir,'rendered videos'))

img_data, coordinates, training_pairs, img_h, img_w = load_data(args)

use_gpu = torch.cuda.is_available()

model = Net(img_h, img_w, args)
if(use_gpu):
    model = model.cuda()
    print('<Net> Use GPU')
else:
    print('<Net> Use CPU')

model.load_state_dict(torch.load(os.path.join(args.savedir,'trained model/trained_model.pt')))
model.eval()

def gen_video():

    precomputed_flows = []

    for i in range(len(coordinates)):
        flows_out = model(torch.from_numpy(coordinates[[i],::]).permute(0,3,1,2).float().cuda(), flow=True)
        precomputed_flows.append(flows_out[0,::].cpu().detach().numpy())
      
    precomputed_flows = np.stack(precomputed_flows,0)

    print(precomputed_flows.shape)

    print('<Gen Video> Number of neighbors for interpolation: {}'.format(num_n))

    if(args.type == ['light','view','time']):
    
        max_light = dims[0]-1
        max_view = dims[1]-1
        max_time = dims[2]-1

        X_light = np.linspace(0, max_light, max_light * scale)
        X_light = np.append(X_light, np.flip(X_light))
        X_view = np.linspace(0, max_view, max_view * scale)
        X_view = np.append(X_view, np.flip(X_view))
        X_time = np.linspace(0, max_time, max_time * scale)
        X_time = np.append(X_time, np.flip(X_time))
        X_light_center = max_light * 0.5 * np.ones(X_light.shape)
        X_view_center = max_view * 0.5 * np.ones(X_view.shape)
        X_time_center = max_time * 0.5 * np.ones(X_time.shape)

        all_dimensions = {
            'light': np.stack([X_light, X_view_center, X_time_center], 1),
            'view': np.stack([X_light_center, X_view, X_time_center], 1),
            'time': np.stack([X_light_center, X_view_center, X_time], 1),
            'light_view': np.stack([X_light, X_view, X_time_center], 1),
            'light_time': np.stack([X_light, X_view_center, X_time], 1),
            'view_time': np.stack([X_light_center, X_view, X_time], 1),
            'light_view_time': np.stack([X_light, X_view, X_time], 1)
        }

        for case, idx in all_dimensions.items():

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('{}/rendered videos/rendered_{}.mp4'.format(args.savedir, case), fourcc, fps, (img_w, img_h))

            print('--------- interpolating {} ---------'.format(case))

            for i in range(len(idx)):

                input_coord = np.array([[[idx[i,:]]]])
                # find the num_n coords that are cloest to the input_coord
                indices = np.argsort(np.sum(np.square(input_coord[0,0,0,:]-coordinates[:,0,0,:]), -1))[:num_n]

                neighbor_coord = coordinates[indices,::]

                input_neighbors = img_data[indices,::]
                input_flows = precomputed_flows[indices,::]

                time_index = indices // (dims[0]*dims[1])
                rest = indices % (dims[0]*dims[1])
                view_index = rest % dims[1]
                albedo_index = view_index*dims[1] + time_index

                img = model(
                    torch.from_numpy(input_coord).cuda().float().permute(0,3,1,2), 
                    neighbors=torch.from_numpy(input_neighbors).cuda().float().permute(0,3,1,2),
                    albedo_index=torch.from_numpy(albedo_index).cuda().long(),
                    coord_neighbor=torch.from_numpy(neighbor_coord).cuda().permute(0,3,1,2), 
                    neighbors_flow=torch.from_numpy(input_flows).cuda(), 
                    test=True
                )

                img = img.cpu().detach()[0,::].permute(1,2,0).numpy()
                img = np.minimum(np.maximum(img, 0.0), 1.0)
                out.write(np.uint8(img * 255))
                print('\r<Gen Video> interpolated image {} of {}'.format(i+1, len(idx)), end=' ')

            out.release()
            print('')
        
    else:
        # args.neighbor_num must be 2
        max_coord = dims[0]-1

        X_coord = np.linspace(0, max_coord, max_coord * scale)
        X_coord = np.append(X_coord, np.flip(X_coord))
        
        all_dimensions = {
            args.type[0]: np.stack([X_coord], 1),
        }

        for case, idx in all_dimensions.items():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('{}/rendered videos/rendered_{}.mp4'.format(args.savedir, case), fourcc, fps, (img_w, img_h))

            print('--------- interpolating {} ---------'.format(case))

            for i in range(len(idx)):
                import math
                input_coord = np.array([[[idx[i,:]]]])
                indices = np.argsort(np.sum(np.square(input_coord[0,0,0,:]-coordinates[:,0,0,:]), -1))[:num_n]
                neighbor_coord = coordinates[indices,::]

                input_neighbors = img_data[indices,::]
                input_flows = precomputed_flows[indices,::]

                img = model(
                    torch.from_numpy(input_coord).cuda().float().permute(0,3,1,2), 
                    neighbors=torch.from_numpy(input_neighbors).cuda().float().permute(0,3,1,2),
                    coord_neighbor=torch.from_numpy(neighbor_coord).cuda().permute(0,3,1,2), 
                    neighbors_flow=torch.from_numpy(input_flows).cuda(), 
                    test=True
                )

                img = img.cpu().detach()[0,::].permute(1,2,0).numpy()
                img = np.minimum(np.maximum(img, 0.0), 1.0)
                out.write(np.uint8(img * 255))
                print('\r<Gen Video> interpolated image {} of {}'.format(i+1, len(idx)), end=' ')

            out.release()
            print('')


gen_video()