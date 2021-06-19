import numpy as np
import cv2
import os

def load_data(args):

    image_filenames = os.listdir(args.dataset)

    img = cv2.imread(os.path.join(args.dataset, image_filenames[0]))
    img_h, img_w, _ = img.shape
    img_h = img_h // args.DSfactor
    img_w = img_w // args.DSfactor
    neighbor_num = args.neighbor_num # unused when dims = 3
    dims = args.dims

    print('<Load Data> Resolution after Downsampling: %d x %d' % (img_h, img_w))

    coordinates = []
    pairs = []
    img_data = []

    # Only handles args.type == ['light', 'view', 'time']
    for i in range(len(image_filenames)):

        print('\r<Load Data> Loading image No.{}'.format(i + 1), end=' ')
        
        img = cv2.imread(os.path.join(args.dataset, image_filenames[i]))
        # when downsampling images, recommend using INTER_AREA as interpolation
        img = cv2.resize(img, dsize=(img_w, img_h), interpolation=cv2.INTER_AREA)
        
        img = np.float32(img) / 255.0   

        time  = i // (dims[0]*dims[1])
        rest  = i %  (dims[0]*dims[1])
        view  = rest % dims[1]
        light = rest // dims[1]

        coordinates.append(np.array([[[light,view,time]]]))
        img_data.append(img)

        pair = np.array([light,light-1,light+1,view,view,view,time,time,time])
        pair = np.where(pair < 0, dims[0]-1, pair)
        pair = np.where(pair > dims[0]-1 , 0, pair)
        pairs.append(pair)
           
           
        pair = np.array([light,light,light,view,view-1,view+1,time,time,time])
        pair = np.where(pair < 0, dims[1]-1, pair)
        pair = np.where(pair > dims[1]-1 , 0, pair)
        pairs.append(pair)
           
           
        pair = np.array([light,light,light,view,view,view,time,time-1,time+1])
        pair = np.where(pair < 0, dims[2]-1, pair)
        pair = np.where(pair > dims[2]-1 , 0, pair)
        pairs.append(pair)
    
    pairs = np.stack(pairs,0)
    img_index    = pairs[:,0:3]*dims[1] + pairs[:,3:6] + pairs[:,6:9]*dims[0]*dims[1] 
    albedo_index = pairs[:,3:4]*dims[1] + pairs[:,6:7]
    training_pairs = np.concatenate((img_index,albedo_index),-1)
    img_data = np.stack(img_data,0)
    coordinates = np.stack(coordinates,0)

    # img_data -> (img_count, h, w, RGB)
    # coordinates -> (img_count, 1, 1, coords), coords as in [light, view, time]
    # training_pairs -> (img_count * dims, img_indexes), img_indexes as in [self, left_neighbor, right_neighbor, albedo]

    print('\n<Load Data> Finish Loading data')

    return img_data, coordinates, training_pairs, img_h, img_w