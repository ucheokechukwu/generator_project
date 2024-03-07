
def plot_training(model, savefig_loc='training.png'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(model.history.history['val_accuracy'], color='green')
    ax[0].set_title('Validation Accuracy')


    ax[1].plot(model.history.history['val_loss'], color='red')
    ax[1].set_title('Validation Loss')


    ax[2].plot(model.history.history['accuracy'], color='blue')
    ax[2].set_title('Training Accuracy')
    plt.xlabel('epochs')
    plt.tight_layout()
    plt.savefig(f'images/{savefig_loc}')
    plt.show()
