
import torch
from lib.model.motion import MDFM
import sys


class X_input(torch.nn.Module):
    """
    combines input video segments with audio and add motion modality if specified
    MDFM() - EXTRACT MOTION -
    """
    def __init__(self, param_TSM ):
        super(X_input, self).__init__()

        self.input_mode = param_TSM["main"]["input_mode"]
        self.n_video_segments = param_TSM["video_segments"] #[8, 8, 1]
        self.n_audio_segments = param_TSM["audio_segments"]  # [8, 8, 1]
        self.n_motion_segments = 0

        if param_TSM["motion"]:
            self.MDF = MDFM(param_TSM["motion_param"])
            self.n_motion_segments = self.n_video_segments


    def forward(self, x_video, x_audio):

        with torch.no_grad():

            if self.n_motion_segments > 0:
                xMOT = self.MDF(x_video)

                if self.n_video_segments > 0:
                    k_frames = x_video.size()[3]
                    if k_frames > 1:
                        xRGB = x_video[:, :, :, k_frames // 2]
                    else:
                        xRGB = x_video

                    if self.input_mode  == 1:
                        x = torch.cat((xRGB, xMOT), dim=2)

                    elif self.input_mode == 2:
                        x = torch.stack([torch.stack([xRGB[:, :, i], xMOT[:, :, i]]) for i in range(self.n_video_segments)])
                        x = x.view((-1,) + (x.size()[-5:])).permute((1, 2, 0, 3, 4, 5))

                    else:
                        sys.exit(f'VCM_input: incorrect mode: {self.input_mode}')
                else:
                    x = xMOT

            elif self.n_video_segments  > 0:
                x = torch.squeeze(x_video, -4)


            if self.n_audio_segments > 0:
                if self.n_video_segments > 0:
                    x = torch.cat((x, x_audio), dim=2)
                else:
                    x = x_audio

        return x




