# AI feels emotions
Can AI feels emotions? We do not know the answer. But AI defenetly can understand what emotions would humans feel while watching a video.


# What is this?
This is supposed to be trained Neural Network for annotation of a video with emotions it (video) would evoke in humans while watching the video.
Unfortunately I am not able to provide trained network weights as this is IP of the company. 

If you run this code on your video the annotation would be done using random weigths.

However I put several examples of video annotated with trained Neural Network to give you idea for the quality of annotation.



## Overview
The Neural Network was trained on 30,751 video ads each mannually annotated by 75 persons with 8 emotion they have expirianced while watching the video.
video ads. More details see [here](Adcumen.pdf).

# How to use it
To annotate video with emotions it would evoked in the watching auditory run:

```bash
# test Something-Something V1
python3 annotate_video.py <path_video_file>
  
```
The result would be stored in "output" folder in subfolder with numeric id.   

 

## Example of videos annotated with trained network 

To demonstrate ability of the trained to recognise evoked emotion by video I put several examples here.

We aimed to create 2 groups of videos known to evoke significant emotion response. 
One group is supposed to invoke positive emotions (like Happiness or Surprise) while the opposite group is known to evoke significant negative ones (fear, anger, sadness). 


We selected top rated videos by YouTube searching engine associated with searching string "heartbreaking moments in movies part". 
Most of the videos from this category represent a sequence of short scenes from different movies evoking sadness, anger or fear. 
As counter example, we selected top rated videos using the search string "animals reunited with owners". 
Most of these videos are a compilation of short clips with animals reunited with their owners and obviously evoking Happiness.









