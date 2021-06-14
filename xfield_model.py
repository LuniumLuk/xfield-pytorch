import torch
import torch.nn as nn
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, img_h, img_w, args):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.args = args

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
        shading = flag*neighbors/albedo + (1-flag)*neighbors 

        x_base, y_base = torch.meshgrid(torch.linspace(-1.0, 1.0, self.img_h), torch.linspace(-1.0, 1.0, self.img_w))
        grid_base = torch.stack((y_base, x_base)).permute(1,2,0).unsqueeze(0).repeat(2,1,1,1).cuda()

        offset_forward_grid = grid_base + offset_forward.permute(0,2,3,1)

        warped_shading = torch.nn.functional.grid_sample(shading.float(), offset_forward_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_view_flow = torch.nn.functional.grid_sample(view_flow_neighbor, offset_forward_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_time_flow = torch.nn.functional.grid_sample(time_flow_neighbor, offset_forward_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_light_flow = torch.nn.functional.grid_sample(light_flow_neighbor, offset_forward_grid, mode='bilinear', padding_mode='border', align_corners=False)

        warped_image = flag*warped_shading*albedo + (1-flag)*warped_shading

        offset_backward = delta_light*warped_light_flow + delta_view*warped_view_flow + delta_time*warped_time_flow

        # Handeling Consistency
        dist = torch.sum(torch.abs(offset_forward-offset_backward), dim=1, keepdim=True)
        weight = torch.exp(-self.args.sigma*img_w*dist)
        weight_normalized = weight/(torch.sum(weight,dim=0,keepdim=True) + epsilon)
        interpolated = torch.sum(torch.multiply(warped_image, weight_normalized), dim=0, keepdim=True)

        return interpolated

    def blending_test(self, coord_in, coord_neighbor, neighbors_img, neighbors_flow, flows, albedo):
        
        epsilon = 0.00001

        img_h = self.img_h
        img_w = self.img_w

        light_flow = flows[:1,0:2,:,:]
        view_flow = flows[:1,2:4,:,:]
        time_flow = flows[:1,4:6,:,:]

        light_flow_neighbor = neighbors_flow[:,0:2,:,:]
        view_flow_neighbor = neighbors_flow[:,2:4,:,:]
        time_flow_neighbor = neighbors_flow[:,4:6,:,:]

        delta = torch.tile(coord_in - coord_neighbor, (1, 1, img_h, img_w))
        delta_light = delta[:,0:1,:,:]
        delta_view = delta[:,1:2,:,:]
        delta_time = delta[:,2:3,:,:]

        forward_shading = delta_view*view_flow + delta_time*time_flow + delta_light*light_flow
        forward_albedo = delta_view*view_flow + delta_time*time_flow
        shading = neighbors_img / albedo

        x_base, y_base = torch.meshgrid(torch.linspace(-1.0, 1.0, self.img_h), torch.linspace(-1.0, 1.0, self.img_w))
        grid_base = torch.stack((y_base, x_base)).permute(1,2,0).unsqueeze(0).repeat(forward_shading.shape[0],1,1,1).cuda()

        forward_shading_grid = grid_base + forward_shading.permute(0,2,3,1)
        forward_albedo_grid = grid_base + forward_albedo.permute(0,2,3,1)

        warped_shading = torch.nn.functional.grid_sample(shading.float(), forward_shading_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_view_flow = torch.nn.functional.grid_sample(view_flow_neighbor, forward_shading_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_time_flow = torch.nn.functional.grid_sample(time_flow_neighbor, forward_shading_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_light_flow = torch.nn.functional.grid_sample(light_flow_neighbor, forward_shading_grid, mode='bilinear', padding_mode='border', align_corners=False)
        warped_albedo = torch.nn.functional.grid_sample(albedo.float(), forward_albedo_grid, mode='bilinear', padding_mode='border', align_corners=False)

        backward_shading = delta_view*warped_view_flow + delta_time*warped_time_flow + delta_light*warped_light_flow 
        backward_albedo = delta_view*warped_view_flow + delta_time*warped_time_flow

        # Handeling Consistency
        dist_shading = torch.sum(torch.abs(backward_shading-forward_shading), dim=1, keepdim=True)
        weight_shading = torch.exp(-self.args.sigma*img_w*dist_shading)
        weight_occ_shading = weight_shading / (torch.sum(weight_shading,0,keepdim=True) + epsilon)
        multiplied = torch.multiply(warped_shading,weight_occ_shading)
        novel_shading = torch.sum(multiplied,0,keepdim=True)

        dist_albedo = torch.sum(torch.abs(backward_albedo-forward_albedo), dim=1, keepdim=True)
        weight_albedo = torch.exp(-self.args.sigma*img_w*dist_albedo)
        weight_occ_albedo = weight_albedo / (torch.sum(weight_albedo,0,keepdim=True) + epsilon)
        multiplied = torch.multiply(warped_albedo,weight_occ_albedo)
        novel_albedo = torch.sum(multiplied,0,keepdim=True)

        interpolated = novel_shading * novel_albedo

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

    def forward(self, x, neighbors=None, albedo_index=None, coord_neighbor=None, neighbors_flow=None,  flow=False, test=False):

        input_x = torch.clone(x)

        for i in range(self.layer_num):
            x = self.nets[i * 4](x)
            x = self.nets[i * 4 + 1](x)
            x = self.nets[i * 4 + 2](x)
            x = self.nets[i * 4 + 3](x)
            if(i == 0):
                coordconv_tl = torch.tile(self.coordconv, [x.shape[0],1,1,1])
                coordconv_tl = self.coord_pad(coordconv_tl)
                x = torch.cat((x, coordconv_tl), dim=1)
        
        x = self.flow_pad(x)
        x = self.flow_conv2d(x)
        x = self.flow_activate(x)

        if(flow): 
            return x
        
        if(not test and not isinstance(albedo_index, np.int32)):
            albedo_index = albedo_index.int().item()
        
        albedo = self.albedos[albedo_index,::]
        if(test):
            interpolated = self.blending_test(
                input_x, 
                coord_neighbor, 
                neighbors, 
                neighbors_flow, 
                x, 
                albedo)
        else:
            interpolated = self.blending(input_x, neighbors, x, albedo)
        
        if(test):
            return interpolated
        else:
            return interpolated, x