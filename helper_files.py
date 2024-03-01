
def plot_training(model):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(model.history.history['val_accuracy'], color='green')
    ax[0].set_title('Accuracy')


    ax[1].plot(model.history.history['val_loss'], color='blue')
    ax[1].set_title('Loss')
    plt.xlabel('epochs')
    plt.tight_layout()
    plt.show()
