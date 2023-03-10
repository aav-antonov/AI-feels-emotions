
import torch
from torch import nn
from torch.nn.init import normal_, constant_
import torchvision

import sys
import timm


from lib.model.temporal_fusion import make_Shift


def timm_load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key: continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model





class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class BackBone(nn.Module):
    def __init__(self, param_TSM):
        super(BackBone, self).__init__()

        self.input_mode = param_TSM["main"]["input_mode"]
        self.n_video_segments = param_TSM["video_segments"]  # [8, 8, 1]
        self.n_audio_segments = param_TSM["audio_segments"]  # [8, 8, 1]
        self.n_motion_segments = 0
        if param_TSM["motion"]: self.n_motion_segments = self.n_video_segments

        self.param = param_TSM
        #print("BackBone param:\n", self.param)

        self.base_model = self.prepare_base_model(param_TSM["main"])
        self.base_model = self.insert_shift_temporal(self.base_model, param_TSM["shift_temporal"])
        self.base_model = self.insert_shift_temporal_modality(self.base_model, param_TSM["shift_temporal_modality"])

        self.features_dim_out = getattr(self.base_model, self.base_model.last_layer_name).in_features

        #print("BackBone output_dim:", self.features_dim_out)


        setattr(self.base_model, self.base_model.last_layer_name, Identity())

        self.dropout_last = nn.Dropout(p=param_TSM['main']["dropout"])

        if param_TSM['main']["last_pool"] > 1:
            self.lastpool = torch.nn.MaxPool1d(param_TSM['main']["last_pool"])
            self.features_dim_out = self.features_dim_out // param_TSM['main']["last_pool"]
            #print("BackBone output_dim last_pool adjusted:", self.features_dim_out)






    def prepare_base_model(self, param):

        if "timm"  in param["arch"]:
            base_model = timm.create_model('resnet50', pretrained=False)
            #base_model = timm_load_model_weights(base_model, "net_weigths/resnet50_miil_21k.pth")
            print("prepare_base_model timm, random weights")
        else:
            base_model = getattr(torchvision.models, param["arch"])(True if param["pretrain"] == 'imagenet' else False)

        if 'resnet' in param["arch"]:
            base_model.last_layer_name = 'fc'
        else:
            raise ValueError(f'Unknown BackBone base model: {param["arch"]}')

        return base_model

    def insert_shift_temporal(self, base_model, param):
        status = param["status"]
        f_div = param["f_div"]
        shift_depth = param["shift_depth"]
        n_insert = param["n_insert"]
        m_insert = param["m_insert"]

        n_video_segments, n_motion_segments, n_audio_segments = self.n_video_segments, self.n_motion_segments, self.n_audio_segments
        input_mode = self.input_mode
        mode = "shift_temporal"


        #print("insert_temporal_shift", param)


        if status:

            #print("insert_temporal_shift n_video_segments, n_motion_segments n_audio_segments", n_video_segments, n_motion_segments, n_audio_segments)
            #print(f"make_temporal_shift n_insert={n_insert} m_insert={m_insert} f_div={f_div} input_mode={input_mode}")
            make_Shift(base_model, n_video_segments, n_motion_segments, n_audio_segments, input_mode, f_div, shift_depth, mode, n_insert, m_insert )

        return base_model

    def insert_shift_temporal_modality(self, base_model, param):

        status = param["status"]
        f_div = param["f_div"]
        n_insert = param["n_insert"]
        m_insert = param["m_insert"]
        n_video_segments, n_motion_segments, n_audio_segments = self.n_video_segments, self.n_motion_segments, self.n_audio_segments
        input_mode = self.input_mode
        mode = "shift_temporal_modality"

        if n_video_segments < 1: status = False
        if n_motion_segments < 1: status = False

        if status:
            #print("insert_modality_shift status", status)
            #print(f"n_insert={n_insert} m_insert={m_insert} f_div={f_div} input_mode={input_mode}")
            shift_depth = 1 # not involved
            make_Shift(base_model,  n_video_segments, n_motion_segments, n_audio_segments, input_mode, f_div, shift_depth, mode, n_insert, m_insert )

        return base_model


    def forward(self, x ):

        n_samples  = x.size()[1]
        n_segs     = x.size()[2]
        x = x.reshape((-1,) + x.size()[-3:])
        x = self.base_model(x)
        x = self.dropout_last(x)

        if self.param['main']["last_pool"] > 1:
            x = torch.unsqueeze(x, 1)
            x = self.lastpool(x)
            x = torch.squeeze(x, 1)

        x = x.view((-1,) +(n_samples, n_segs) + x.size()[-1:])
        return x



