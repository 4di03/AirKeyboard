import cv2
import os
import json
import numpy as np
import pandas as pd
import sys
"""
def example_show_keypoints(args, sid, fid, cid):
    # load image
    image_file = os.path.join(args.hanco_path, f'rgb/{sid:04d}/cam{cid}/{fid:08d}.jpg')
    img = cv2.imread(image_file)[:, :, ::-1]

    # load keypoints
    kp_data_file = os.path.join(args.hanco_path, f'xyz/{sid:04d}/{fid:08d}.json')
    with open(kp_data_file, 'r') as fi:
        kp_xyz = np.array(json.load(fi))
    print('kp_xyz', kp_xyz.shape, kp_xyz.dtype)  # 21x3, np.float64, world coordinates
    # load calibration
    calib_file = os.path.join(args.hanco_path, f'calib/{sid:04d}/{fid:08d}.json')
    with open(calib_file, 'r') as fi:
        calib = json.load(fi)

"""


def keypoint_data_to_csv(data_path= "/Users/adithyapalle/work/CS5100/Project/data/hanco_all/HanCo_tester"):
    """
    Creates a csv mapping image file names to the corresponding json files containingkeypoint data 
    """
    data_path = os.path.abspath(data_path)
    train_out_loc = os.path.join(data_path, "train_keypoint_data.csv")
    test_out_loc = os.path.join(data_path, "test_keypoint_data.csv")

    rgb_path = os.path.join(data_path, "rgb")
    meta_path = os.path.join(data_path, "meta.json")
    meta = json.load(open(meta_path, "r"))

    i = 0
    #print(len(meta["is_train"][i]),len(meta["subject_id"][i]))
    #return
    train_data = []
    test_data = []

    for dir in sorted(os.listdir(rgb_path)):
        sid = int(dir)
        for cam in sorted(os.listdir(os.path.join(rgb_path, dir))):
            j = 0
            cid = cam
            jpg_files = sorted(os.listdir(os.path.join(rgb_path, dir, cam)))
            for file in jpg_files:
                fid = int(file.split(".")[0])
                train_flag = meta["is_train"][sid][fid]

                #id_pairs.append((sid, fid, cid))
                image_file = os.path.join(data_path, f'rgb/{sid:04d}/{cid}/{fid:08d}.jpg')

                kp_data_file = os.path.join(data_path, f'xyz/{sid:04d}/{fid:08d}.json')

                calib_file = os.path.join(data_path, f'calib/{sid:04d}/{fid:08d}.json')
                if train_flag:
                    train_data.append((image_file, kp_data_file, calib_file))
                else:
                    test_data.append((image_file, kp_data_file, calib_file))
                j += 1
        
        i += 1


    train_df = pd.DataFrame(train_data, columns=["image_file", "kp_data_file", "calib_file"])
    test_df = pd.DataFrame(test_data, columns=["image_file", "kp_data_file", "calib_file"])

    train_df.to_csv(train_out_loc, index=False)

    test_df.to_csv(test_out_loc, index=False)

    

if __name__ == "__main__":

    keypoint_data_to_csv(data_path = sys.argv[1])