
import matplotlib.pyplot as plt
import pandas as pd



def plot_loss_curve(loss_path,title = "Loss Curve"):
    df = pd.read_csv(loss_path)
    
    get_loss = lambda s: float(s.split("=")[1])
        
    train_losses = df.iloc[:, 1]

    #print(train_losses,type(train_losses[2].split("=")[1])

    train_losses = list(map(get_loss, train_losses))
    val_losses = df.iloc[:, 2]
    val_losses = list(map(get_loss,val_losses))


    epochs = range(len(train_losses))
    
    
    plt.plot(epochs, train_losses, label = "Train")
    plt.plot(epochs, val_losses, label = "Validation")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title(title)
    plt.legend()
    plt.show()