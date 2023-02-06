import os
import argparse
import json
import random


import os, sys
import torch
import torchaudio
from collections import Counter
import sys

from lib.model.model import VCM
from lib.model.prepare_input import X_input
from lib.model.backbone import BackBone

from lib.dataset.audio import wafeform_to_spectrogram_torch
from lib.dataset.video import  load_image_f, validation_transform
from lib.video.indicator import IndicatorOnImage


def makedir(path):
    if os.path.exists(path):
        print("path exist: ", path)
    else:
        try:
            os.mkdir(path)
        except:
            raise Exception("Can not create folder: {path}")
            sys.exit(1)

def convert_video_to_frames(file_video, folder4tmp, id_run, fps):
    if id_run != None:
        job_id = id_run
    else:
        job_id = len(os.listdir(folder4tmp))

    output_folder = f'{folder4tmp}/{job_id}'
    output_folder_image = f'{output_folder}/image'
    makedir(output_folder)
    makedir(output_folder_image)

    cmd_frames = f'ffmpeg -loglevel panic -i {file_video}  -vf \"fps={fps}\" -q:v 0 \"{output_folder_image}/%05d.jpg\" '

    try:
        os.system(cmd_frames)
    except:
        print(f"An exception occurred while converting to frames file: {file_video}\n")
        sys.exit(1)

    cmd_audio = f"ffmpeg -loglevel panic -i {file_video}   {output_folder}/audio.wav"

    try:
        os.system(cmd_audio)
    except:
        print(f"An exception occurred while converting to audio file: {file_video}\n")
        sys.exit(1)


    return output_folder

def set_model_DataParallel( args, model ):

    DataParallel = False
    cuda_ids = args["cuda_ids"]
    if len(cuda_ids) > 1:

        print(f"Let's use {cuda_ids} GPUs out of {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model, device_ids=cuda_ids , output_device=cuda_ids[0], dim=0)
        DataParallel = True

    return model, DataParallel

def load_model(path_checkpoint, model, DataParallel=False, Filter_layers=None):
    if os.path.isfile(path_checkpoint):
        print(f"=> loading checkpoint {path_checkpoint}\n")
        checkpoint = torch.load(path_checkpoint, map_location='cuda:0')
        model_state = checkpoint['model_state']


        if DataParallel:
            for name in model.module.model_layers:
                if name in Filter_layers: continue
                model.module.model_layers[name].load_state_dict(model_state[name])
        else:
            for name in model.model_layers:
                if name in Filter_layers: continue
                model.model_layers[name].load_state_dict(model_state[name])

    else:
        print(f"=> no checkpoint found at {path_checkpoint}")

    return model

def get_model(args):

    model_X = X_input(args["TSM"])
    model_BB = BackBone(args["TSM"])
    model = VCM(args["TSM"], model_X, model_BB)

    """set_cuda_device"""
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
    DataParallel = False
    model.to(device)

    print("device, device_id", device)
    print("DataParallel", DataParallel)



    return model, DataParallel,device

def get_audio( file_audio):
    waveform, sr = torchaudio.load(file_audio)
    return waveform, sr

def get_audio_x(t_start, waveform, sr, args):
    clip_length = args['emotion_jumps']['clip_length']
    audio_img_param = args['dataset']['audio_img_param']

    # cut waveform from interval from t_start: t_start +clip_length
    waveform = waveform[:,sr*t_start : sr*t_start+sr*clip_length]

    X_mel = wafeform_to_spectrogram_torch(waveform[0], sr, audio_img_param)
    X = torch.squeeze(X_mel,0)
    X = torch.unsqueeze(X, 0)
    return X

def predict( model, device,  folder4video_processing,args ):

    clip_length = args['emotion_jumps']['clip_length']
    video_segments = args['TSM']['video_segments']
    fps = args['fps']

    img_input_size  = args['dataset']['video_img_param']['img_input_size']
    img_output_size = args['dataset']['video_img_param']['img_output_size']

    file_audio = f'{folder4video_processing}/audio.wav'
    waveform, sr = get_audio(file_audio)

    folder_images = f'{folder4video_processing}/image'
    list_file_frames = sorted(os.listdir(folder_images))

    video_duration = len(list_file_frames)/fps
    print("video_duration", video_duration)

    # y_pred - predicted lables
    y_pred = []
    for t_start in range(int(video_duration+1-clip_length)):
        sampled_frames = sorted(random.sample(list_file_frames[t_start*fps:(t_start+clip_length)*fps], video_segments))

        x_video = torch.stack([load_image_f(folder_images,ff) for ff in sampled_frames])
        x_video = validation_transform(x_video, img_input_size, img_output_size)
        x_video = torch.unsqueeze(x_video, dim=1)
        x_video = torch.unsqueeze(x_video, 0)
        x_video = torch.unsqueeze(x_video, 0)

        x_audio = get_audio_x(t_start, waveform, sr, args)
        x_audio = torch.unsqueeze(x_audio, 0)
        x_audio = torch.unsqueeze(x_audio, 0)

        x_video = x_video.to(device)
        x_audio = x_audio.to(device)

        output = model(x_video, x_audio)

        _, y_pred_max = torch.max(output.data.cpu(), 1)
        y_pred.extend(y_pred_max.cpu().tolist())


    return y_pred



def main():

    # check input
    if(len(sys.argv) < 2):
        print("Total arguments passed:", len(sys.argv))
        print("Usage:", sys.argv[0] ," <path_to_file.mp4> <path_to_tmp_folder (optional)>\n")
        sys.exit(1)

    # Set folder for video proceccing (framing, annotating and assembling output video)
    tmp_folder = "output"
    #tmp_folder = "youtube_examples"
    makedir(tmp_folder)

    id_run = None
    if(len(sys.argv) == 3):
        id_run = sys.argv[2]
    elif(len(sys.argv) > 3):
        print("Warning: Usage:", sys.argv[0], " <path_to_file.mp4> <path_to_tmp_folder (optional)>\n")
        print("Ignoring arguments ", sys.argv[3:])

    #Set fps parameter (Frame Per Second), Convert input video into frames and audio files
    fps = 10
    file_video = sys.argv[1]
    output_folder = convert_video_to_frames(file_video, tmp_folder, id_run, fps)

    #load args with model parameters
    with open("checkpoint/args.json") as f:
        args = json.load(f)

    #args['emotion_jumps']['clip_length'] = 3
    args['fps'] = fps

    #set model with parameters specified in args
    model, DataParallel,device = get_model(args)

    # load model pretrained weigts
    path_checkpoint = "checkpoint/balanced.ckpt.pth.tar"
    if os.path.exists(path_checkpoint):
        model = load_model(path_checkpoint, model, DataParallel=DataParallel, Filter_layers={})
    else:
        print("No pretrained weights are provided, continue with random weights")

    """ Emotion used for training """
    EmotionNames = {0: "Anger",
                    1: "Contempt",
                    2: "Disgust",
                    3: "Fear",
                    4: "Happiness",
                    5: "Neutral",
                    6: "Sadness",
                    7: "Surprise"
                }

    # predicting emotion labels for 5 seconds intervals:
    #  for t = 0, t < video_duration-5:
    #  emotion_labels = model(video[t:t+5]), where video[t:t+5]-> 16 randomly sampled frames from [t:t+5] + audi[t:t+5]
    y_pred = predict(model, device, output_folder, args )

    # get Freq Distribution (of  each emotion was predicted in the video)
    statW = Counter(y_pred)
    statWE = {EmotionNames[e]:statW[e] for e in statW}

    file_emotion_stat = f'{output_folder}/emotions.json\n'
    with open(file_emotion_stat, 'w') as f:
        json.dump(statWE, f, indent=4)


    clip_length = args['emotion_jumps']['clip_length']
    sliding_window = []
    sliding_window_length = 5
    IonI = IndicatorOnImage(EmotionNames, value_max=sliding_window_length)

    for i,l in enumerate(y_pred):
        # add incoming label to sliding_window
        sliding_window.append(l)

        # remove the last label  from sliding_window
        if len(sliding_window) > sliding_window_length: del sliding_window[0]

        # init stat
        stat = {v:0 for v in range(8)}

        for v in sliding_window:
            if v in range(8): stat[v] += 1

        # get frames in the 1 second interval to current
        frame_s, frame_e = int((i + clip_length / 2) * fps), int(((i + clip_length / 2) + 1) * fps)

        # insert Emotion Indicator Bar for frames in range(frame_s, frame_e)
        for f in range(frame_s, frame_e):
            f_str = str(f + 1).zfill(5)  # format 102 -> 000102
            file_image = f"{output_folder}/image/{f_str}.jpg"

            #insert Emotion Indicator Bar into frame (file_image)
            if os.path.isfile(file_image):
                IonI.add_on_image(file_image, stat, output_path=file_image)

    # assemble output video with modified frames
    ## -vcodec libx265 add this to compress the video better
    cmd_assemble = f"ffmpeg -loglevel panic -framerate {fps} -i \"{output_folder}/image/%05d.jpg\" -i {output_folder}/audio.wav  -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \"{output_folder}/output.mp4\" "
    try:
        os.system(cmd_assemble)
    except:
        print(f"An exception occurred \n")

    print("Finished")
    print(f"Annotated video: {output_folder}/output.mp4")
    print(f"Emotion statistics: {output_folder}/emotions.json")

if __name__ == '__main__':
    main()