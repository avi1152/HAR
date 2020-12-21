# HAR ####(Anmol Paliwal, Avinash Kumar, Sanaka Nadeesh)
Human Activity Recognition

## Downloading the Dataset
Link: https://20bn.com/datasets/something-something
Data Size = 20GB

After Downloading separate the .webm video files in a directory named videos/
And the .json files to annotations/

20bn-something-something-v2 -|---annotations/[All the .json files]
                             |
                             |
                             |---videos/[All the .webm files]
                             
## Modifying Paths
Change the paths for the .json files and .webm files in the main.py configurations dictionary.

## Training From Scratch
Run: CUDA_VISIBLE_DEVICES=0,1 python train.py -c configs/config_model1_224.json -g 0,1 --use_cuda

## Adjust Hyperparameters
In the main.py file,
we can change the value for configurations dictionary as,

clip_size: Number of frames to take from a .webm video file
input_spatial_size: For cropping the each frame into the input_spatial_size
column_units: Number of feature units for each sample
batch_size: Fill depending on the GPU
