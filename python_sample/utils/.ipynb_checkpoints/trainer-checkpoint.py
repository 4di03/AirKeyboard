import numpy as np
import torch


def save_state(model, optimizer, scheduler, train_dataloader, val_dataloader, save_loc):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
    }, save_loc)
    
    

def load_state(model, optimizer, scheduler, train_dataloader, val_dataloader, load_loc):
    checkpoint = torch.load(load_loc)

    # Load model, optimizer, and scheduler states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load dataloaders (assuming they are pickleable)
    loaded_train_dataloader = checkpoint['train_dataloader']
    loaded_val_dataloader = checkpoint['val_dataloader']

    return model, optimizer, scheduler, loaded_train_dataloader, loaded_val_dataloader


class Trainer:
    def __init__(self, model, criterion, optimizer, config, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epochs = config["epochs"]
        self.batches_per_epoch = config["batches_per_epoch"]
        self.batches_per_epoch_val = config["batches_per_epoch_val"]
        self.device = config["device"]
        self.scheduler = scheduler
        self.checkpoint_frequency = 100
        self.early_stopping_epochs = 10
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.save_per_n_epochs = config["save_per_n_epochs"] if "save_per_n_epochs" in config else 10

    def train(self, train_dataloader, val_dataloader, state_loc):
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                )
            )

            # reducing LR if no improvement
            if self.scheduler is not None:
                self.scheduler.step(self.loss["train"][-1])

            # saving model
            if (epoch + 1) % self.checkpoint_frequency == 0:
                torch.save(
                    self.model.state_dict(), "model_{}".format(str(epoch + 1).zfill(3))
                )

            # early stopping
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]), 
                                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0
                    #print('New min: ', min_val_loss)

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break
            if epoch % self.save_per_n_epochs == 0 and state_loc is not None:
                print(f"{epoch} Epochs elapse, checkpointing training state")
                save_state(self.model, self.optimizer, self.scheduler, train_dataloader, val_dataloader, state_loc)

        torch.save(self.model.state_dict(), "model_final")
        return self.model

    def _epoch_train(self, dataloader):
        self.model.train()
        running_loss = []
        

        for i, data in enumerate(dataloader, 0):
#             print("L74: ", i, self.batches_per_epoch)
            inputs = data["image"].to(self.device)
            labels = data["heatmaps"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.batches_per_epoch:
                epoch_loss = np.mean(running_loss)
                self.loss["train"].append(epoch_loss)
                break

    def _epoch_eval(self, dataloader):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                #print("L97: ", i, self.batches_per_epoch_val)
                inputs = data["image"].to(self.device)
                labels = data["heatmaps"].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.batches_per_epoch_val:
                    epoch_loss = np.mean(running_loss)
                    self.loss["val"].append(epoch_loss)
                    break
