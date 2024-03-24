from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import set_random_seed
from data_generation import sample_data
from keras.metrics import Precision, Recall
import numpy as np


set_random_seed(0)


def scheduler(epoch, lr):
    return lr if epoch<=5 else lr*np.exp(-0.1)


def train_and_evaluate_model(model=None, 
                            trainset=None, 
                            patience=5,
                            num_epochs=20,
                            metrics='accuracy',
                            training_verbose=True,
                            **qwargs):

    """
    model: keras model. If none, function returns the train and test data
    trainset: one of ['X0', 'X123', 'X4', 'X6'] or None; 
                can only be None, if it is a submodel;
    qwargs for sample_data () function
    """    
    
    ### data
    
    if model is None and trainset is None:
        return "Insufficient parameters. Include either a model or a trainset."
    elif model is None and trainset not in ['X0', 'X123', 'X4', 'X6']:
        return  "Invalid trainset value."      
    elif model is not None and trainset is None:
        trainset = 'X'+type(model).__name__.split('X')[-1]
    
    data = sample_data(**qwargs)

    dictionary_of_trainsets = {
        'X4': [8, 9],
        'X0': [0, 1],
        'X123': [(slice(2,5)), slice(5,8)],
        'X6': [12, 13],    }
    y_train, y_test = data[-2], data[-1]

    X_train = data[dictionary_of_trainsets[trainset][0]]
    X_test = data[dictionary_of_trainsets[trainset][1]]
    if not model:
        return X_train, X_test, y_train, y_test
    
    
    ### training
    earlystopping = EarlyStopping(patience=patience,
                                      monitor='val_accuracy',
                                      restore_best_weights=True,
                                      verbose=1)
    lr_scheduler = LearningRateScheduler(scheduler)
    

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adamax',
              metrics=metrics)
    try:
        model.fit(X_train, y_train,
          validation_data = (X_test, y_test),
          batch_size=32,
          epochs=num_epochs,
          verbose=int(training_verbose),
          callbacks=[earlystopping, lr_scheduler],
#           steps_per_epoch = 1,
          )
        return model.evaluate(X_test, y_test, verbose=2)[-1], model

    except Exception as e:
        print(f"{model.name} failed to train.\n{str(e)}")
        return -1, model
