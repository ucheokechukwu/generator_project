import matplotlib.pyplot as plt
import seaborn as sns
def plot_training(model, savefig_loc='training.png'):
    
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(model.history.history['val_accuracy'], color='green')
    ax[0].set_title('Validation Accuracy')


    ax[1].plot(model.history.history['val_loss'], color='red')
    ax[1].set_title('Validation Loss')


    ax[2].plot(model.history.history['accuracy'], color='blue')
    ax[2].set_title('Training Accuracy')
    plt.xlabel('epochs')
    plt.tight_layout()
    if savefig_loc != "":
        plt.savefig(f'images/{savefig_loc}')
    plt.show()
    
class bcolors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    BROWN = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
# SAMPLE:
# print(f"{bcolors.UNDERLINE}Warning: No active frommets remain. Continue?{bcolors.ENDC}")   


def confusion_matrix_plotter(m, X1_test, X2_test, X3_test, y_test):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import tensorflow as tf
    
    y_proba = m.predict([X1_test, X2_test, X3_test])
    y_pred = tf.argmax(y_proba, axis=-1)
    y_test.shape, y_pred.shape

    # Assuming y_pred and y_test already exist with shape (1000, 44)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Define custom colormap
    cmap = sns.diverging_palette(120, 20, as_cmap=True)

    # Plot confusion matrix with custom colors and presentation styles
    plt.figure(figsize=(12, 9))  # Adjust the figsize for the desired size
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=np.arange(44), yticklabels=np.arange(44),
                annot_kws={"fontsize": 12, "fontweight": "bold"}, cbar=False, vmin=0, vmax=100)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
