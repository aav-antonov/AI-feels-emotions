
import torch
from torch import nn




class VCM(torch.nn.Module):

    def __init__(self, param_TSM, nn_X_input, nn_backbone ):
        super(VCM, self).__init__()

        self.model_layers = {}

        self.nn_X_input = nn_X_input
        self.model_layers["nn_X_input"] = self.nn_X_input

        self.nn_backbone = nn_backbone
        self.model_layers["nn_backbone"] = self.nn_backbone

        self.num_class = param_TSM["num_class"]
        self.n_video_segments = param_TSM["video_segments"]  # 8
        self.n_audio_segments = param_TSM["audio_segments"]  # 1
        self.n_motion_segments = 0 # 0, or == self.n_video_segments
        if param_TSM["motion"]: self.n_motion_segments = self.n_video_segments

        self.last_num = self.nn_backbone.features_dim_out
        self.last_fc = nn.Linear(self.last_num, self.num_class)
        self.model_layers["last_fc"] = self.last_fc


    def forward(self, x_video, x_audio):

        """combines input video segments with audio and add motion modality if specified"""
        x = self.nn_X_input(x_video, x_audio)

        """feature extraction by resnet50"""
        x = self.nn_backbone(x)

        """ average accross n_sample"""
        x = x.mean(dim=1)


        """ average accross n_seg"""
        x = x.mean(dim=1)

        x = x.view((-1,self.last_num))

        x = self.last_fc(x)

        return x


