import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import sys
sys.path.append("../")

from utils.dataset import FreiHAND, HanCo
from utils.model import ShallowUNet
from utils.trainer import Trainer, save_state, load_state
from utils.prep_utils import (
    blur_heatmaps,
    IoULoss,
    COLORMAP,
    N_KEYPOINTS,
    N_IMG_CHANNELS,
    get_norm_params,
    show_data,
    import_object_from_file
    
)
from torchvision import datasets, transforms






def train(config, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    dataset_type = HanCo#FreiHAND
   
    data_prepped = False




    if not data_prepped:

       # train_dataset = FreiHAND(config=config, set_type="train")
    #     train_dataloader = DataLoader(
    #         train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
    #     )


    #     get_norm_params(train_dataloader)




        train_dataset = dataset_type(config=config, set_type="train")
        train_dataloader = DataLoader(
            train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
        )

        val_dataset = dataset_type(config=config, set_type="val")
        val_dataloader = DataLoader(
            val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
        )


        #data_prepped = True


    print(len(train_dataloader),len(val_dataloader) )

    get_norm_params(train_dataloader)


    # visualize random batch of data train samples + labels
    show_data(train_dataset, n_samples=10)




   # save_loc = "/scratch/palle.a/AirKeyboard/python_sample/weights/untrained_test_model.pt"



    model = model.to(config["device"])

    criterion = IoULoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
    )


    trainer = Trainer(model, criterion, optimizer, config, scheduler)
    model = trainer.train(train_dataloader, val_dataloader, state_loc = config["save_loc"])

    plt.plot(trainer.loss["train"], label="train")
    plt.plot(trainer.loss["val"], label="val")
    plt.legend()
    plt.show()


    # SAVING
    save_loc =  config["save_loc"]
    torch.save(model.state_dict(), save_loc + "_final_save")

if __name__ == "__main__":
    
    cfg_file_path = sys.argv[1]
    
    
    model = import_object_from_file(cfg_file_path, "model")
    config = import_object_from_file(cfg_file_path, "config")
    
    train(config,model)


