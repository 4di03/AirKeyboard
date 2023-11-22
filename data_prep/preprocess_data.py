import pandas as pd
import cv2
import json
def prep_data(csvPath):


    df = pd.read_csv(csvPath)

    for i,row in df.iterrows():

        image_path = row["image_file"]
        keypoint_path = row["kp_data_file"]
        calib_path = row["calib_file"]
        

        image = cv2.imread(image_path)
        raw_keypoints = json.load(open(keypoint_path, "r"))
        calib_matrix = json.load(open(calib_path, "r"))

