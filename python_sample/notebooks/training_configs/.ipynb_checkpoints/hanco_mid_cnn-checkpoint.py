import sys
sys.path.insert(0, '/scratch/palle.a/AirKeyboard/python_sample/utils')
from model import *
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_IMG_CHANNELS = 3
N_KEYPOINTS = 21

model = SimpleCNN(N_IMG_CHANNELS, N_KEYPOINTS)#ShallowUNet(N_IMG_CHANNELS, N_KEYPOINTS)

data_dir = "/scratch/palle.a/AirKeyboard/data/hanco_all/HanCo"#"/scratch/palle.a/AirKeyboard/data/freihand_all/FreiHAND_pub_v2"

config = {
    "data_dir": data_dir,
    "epochs": 1000,
    "batch_size": 48,
    "batches_per_epoch": 100,
    "batches_per_epoch_val":  20,
    "learning_rate": 0.1,
    "device":device ,
    "n_samples" : 60000, 
    "save_loc" : "/scratch/palle.a/AirKeyboard/python_sample/weights/hanco_mid_cnn"
}