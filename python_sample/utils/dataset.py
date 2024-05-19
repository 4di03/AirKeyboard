import numpy as np
import os
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from utils.prep_utils import (
    projectPoints,
    vector_to_heatmaps,
    RAW_IMG_SIZE,
    MODEL_IMG_SIZE,
    DATASET_MEANS,
    DATASET_STDS,
    get_norm_params,
)
import cv2
from tqdm import tqdm
from PIL import UnidentifiedImageError

    


class FreiHAND(Dataset):
    """
    Class to load FreiHAND dataset. Only training part is used here.
    Augmented images are not used, only raw - first 32,560 images

    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config, set_type="train"):
        self.device = config["device"]
        self.image_dir = os.path.join(config["data_dir"], "training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))

        fn_K_matrix = os.path.join(config["data_dir"], "training_K.json")
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))

        fn_anno = os.path.join(config["data_dir"], "training_xyz.json")
        with open(fn_anno, "r") as f:
            self.anno = np.array(json.load(f))
        
        if set_type == "train":
            n_start = 0
            n_end = 26000
        elif set_type == "val":
            n_start = 26000
            n_end = 31000
        else:
            n_start = 31000
            n_end = len(self.anno)
            
        #n_start = 0
        #n_end = 4
        
        #print("L59: ", self.K_matrix.shape)

        self.image_names = self.image_names[n_start:n_end]
        self.K_matrix = self.K_matrix[n_start:n_end]
        self.anno = self.anno[n_start:n_end]

        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            ]
        )

    def __len__(self):
        return len(self.anno)
    
    def __getitem__(self, idx):
#         print("L71 self.image_names: ", self.image_names[idx]) # image name
#         print("self.anno ", self.anno[idx].shape) # raw keypoints ( 21, 3)
#         print("self.K_matrix " , self.K_matrix[idx].shape) # K matrix
        
        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))
        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)
        
        keypoints = projectPoints(self.anno[idx], self.K_matrix[idx])
        keypoints = keypoints / RAW_IMG_SIZE
        heatmaps = vector_to_heatmaps(keypoints)
        keypoints = torch.from_numpy(keypoints)
        heatmaps = torch.from_numpy(np.float32(heatmaps))

        return {
            "image": image,
            "keypoints": keypoints, # keypoints
            "heatmaps": heatmaps,
            "image_name": image_name,
            "image_raw": image_raw,
        }


    
def cv_load(ip):
    img = cv2.imread(ip)#/255
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img/255

def get_cam_index(ip):
    cam_index = None
    for part in ip.split('/'):
        if "cam" in part:
            cam_index = int(part[-1])

    if cam_index is None:
        raise Exception("Couldn't find camera index in file path")
        
    return cam_index
class HanCo(Dataset):
        
    def __init__(self, config, set_type ="train"):
        self.device = config["device"]
        
        set_str = set_type if set_type != "val" else "test"
        #print(set_str)
        
        data_csv_path = os.path.join(config["data_dir"], f"train_keypoint_data.csv")
        
        self.data_csv = pd.read_csv(data_csv_path)
        self.raw_images = []
        self.anno =  []
        self.K_matrix = []
        self.M_matrix = []
        self.ip = []
        self.image_raw_transform = transforms.ToTensor()
        
        raw_transformed_images = []

        self.partial_image_transform = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor()
            ])
        
        self.image_names = []
        printit = False
        
        
        train_end = round(0.8*config["n_samples"])
        val_end = round(0.9*config["n_samples"])
        test_end = min(val_end + 2000, config["n_samples"])
           
        if set_type == "train":
            n_start = 0
            n_end = train_end
        elif set_type == "val":
            n_start = train_end
            n_end = val_end
        else:
            n_start = val_end
            n_end = test_end
        
        rm_ind = []
        
        self.data_csv = self.data_csv.loc[n_start:n_end-1].reset_index(drop = True)
        
        #print("L156: " , len(self.data_csv))
            
        for i in tqdm(range(n_end-n_start)):#range(config["n_samples"])):

            ip = self.data_csv.iloc[i]["image_file"]
    
            
            try:
                image_raw =Image.open(ip)
                
            
                image_raw_trans = self.partial_image_transform(image_raw)# * 255

                    
                raw_transformed_images.append(image_raw_trans)
            except UnidentifiedImageError as e:
                rm_ind.append(i)
                print(f"Couldn't read image {ip}, got error {e}")
                continue
                
#             self.ip.append(ip)
            self.image_names.append(os.path.basename(ip))
            
            self.raw_images.append(image_raw)
            
            kp_path = self.data_csv.iloc[i]["kp_data_file"]
            kp_data = np.array(json.load(open(kp_path,"r")))
            
          #  print("L141: " , kp_data.shape)
            #self.anno.append(kp_data)
        
        
            calib_path = self.data_csv.iloc[i]["calib_file"]
            
            K_data = np.array(json.load(open(calib_path,"r"))["K"])
            M_data = np.array(json.load(open(calib_path,"r"))["M"])
           # print("L152: ", M_data.shape)
            
            

#             K_data = K_data[self.cam_index]
#             M_data = M_data[cam_index]
#            # print("L148", K_data.shape)
#             self.M_matrix.append(M_data)
#             self.K_matrix.append(K_data)
            
        
        stats = get_norm_params(raw_transformed_images,from_tensor = True)
        
        del raw_transformed_images
        
        dset_means = stats["mean"]
        dset_stds = stats["std"]
        
        self.data_csv = self.data_csv.drop(rm_ind)
        
        assert len(self.data_csv) == len(self.raw_images), f"df len : {len(self.data_csv)} , iamges len: {len(self.raw_images)}"

    

#         self.image_paths = self.data_csv["image_file"]
#         self.image_dir = os.path.join(config["data_dir"], "rgb_merged")
#         self.image_names = np.sort(os.listdir(self.image_dir))

#         fn_K_matrix = os.path.join(config["data_dir"], "training_K.json")
#         with open(fn_K_matrix, "r") as f:
#             self.K_matrix = np.array(json.load(f))

#         fn_anno = os.path.join(config["data_dir"], "training_xyz.json")
#         with open(fn_anno, "r") as f:
#             self.anno = np.array(json.load(f))


#         self.image_names = self.image_names[n_start:n_end]
#         self.K_matrix = self.K_matrix[n_start:n_end]
#         self.anno = self.anno[n_start:n_end]
#         self.raw_images = self.raw_images[n_start:n_end]
#         self.M_matrix = self.M_matrix[n_start:n_end]
#         self.ip = self.ip[n_start:n_end]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(MODEL_IMG_SIZE),
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: x * 255),  # Multiply image values by 255
                transforms.Normalize(mean=dset_means, std=dset_stds),
            ]
        )  
        
                    
        
        
    def __len__(self):
        
        return len(self.image_names)
    
    def __getitem__(self, idx):
#         print("L71 self.image_names: ", self.image_names[idx]) # image name
#         print("self.anno ", self.anno[idx].shape) # raw keypoints ( 21, 3)
#         print("self.K_matrix " , self.K_matrix[idx].shape) # K matrix



        kp_path = self.data_csv.iloc[idx]["kp_data_file"]
        kp_data = np.array(json.load(open(kp_path,"r")))

        calib_path = self.data_csv.iloc[idx]["calib_file"]

        ip = self.data_csv.iloc[idx]["image_file"]
        cam_index = get_cam_index(ip)
        K_data = np.array(json.load(open(calib_path,"r"))["K"])[cam_index]
        M_data = np.array(json.load(open(calib_path,"r"))["M"])[cam_index]
        
        
        image_name =  self.image_names[idx]
        image_raw = self.raw_images[idx]

        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)
        
        keypoints = projectPoints(kp_data, K_data, M = M_data)
        keypoints = keypoints / RAW_IMG_SIZE
        
        heatmaps = vector_to_heatmaps(keypoints)
        keypoints = torch.from_numpy(keypoints)
        heatmaps = torch.from_numpy(np.float32(heatmaps))

        return {
            "image": image,
            "keypoints": keypoints, # keypoints
            "heatmaps": heatmaps,
            "image_name": image_name,
            "image_raw": image_raw,
            "ip":ip
        }

