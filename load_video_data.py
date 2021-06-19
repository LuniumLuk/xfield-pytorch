import numpy as np
import cv2
import os
from easydict import EasyDict
# args = EasyDict({
#             'video_path': './data/video/test_cut.mp4',
#             'savedir': './results/3x3x3/apple',
#             'savedir': './results/3x3x3/apple',
#             'type': ['light', 'time', 'view'],
#             'dims': [3, 3, 3],
#             'DSfactor': 8,
#             'neighbor_num': 4,
#             'lr': 0.0001,
#             'sigma': 0.1,
#             'stop_l1_thr': 0.01
#         })

def load_video_data(args):
    cap = cv2.VideoCapture(args.video_path)
    ret, first_frame = cap.read()
    img = first_frame
    img_h, img_w, _ = img.shape
    img_h = img_h // args.DSfactor
    img_w = img_w // args.DSfactor
    dims = args.dims

    print('<Load Video Data> Resolution after Downsampling: %d x %d' % (img_h, img_w))

    # Video Only Handles args.type == ['time']
    coordinates = []
    pairs = []
    img_data = []
    i = 0
    while (1):
        # get a frame
        ret, frame = cap.read()
        if not ret:
            break
        print('\r<Load Data> Loading image No.{}'.format(i + 1), end=' ')
        img = cv2.resize(frame, dsize=(img_w, img_h), interpolation=cv2.INTER_AREA)
        img = np.float32(img) / 255.0
        coordinates.append(np.array([[[i]]]))
        img_data.append(img)

        pair = np.array([i, i - 1, i + 1])
        pair = np.where(pair < 0, dims[0] - 1, pair)
        pair = np.where(pair > dims[0] - 1, 0, pair)
        pairs.append(pair)
        i = i + 1

    img_data = np.stack(img_data, 0)
    coordinates = np.stack(coordinates, 0)
    print(pairs)
    training_pairs = np.stack(pairs, 0)


    print('\n<Load Video Data> Finish Loading Video data')

    return img_data, coordinates, training_pairs, img_h, img_w

args = EasyDict({
            'video_path': './data/video/test_cut.mp4',
            'savedir': './results/3x3x3/apple',
            'type': ['light', 'time', 'view'],
            'dims': [3, 3, 3],
            'DSfactor': 8,
            'neighbor_num': 4,
            'lr': 0.0001,
            'sigma': 0.1,
            'stop_l1_thr': 0.01
        })

load_video_data(args)