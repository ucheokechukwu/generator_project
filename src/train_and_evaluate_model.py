from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import set_random_seed
from data_generation import sample_data


set_random_seed(0)
earlystopping = EarlyStopping(patience=15,
                                      monitor='val_accuracy',
                                      restore_best_weights=True,
                                      verbose=1)

def scheduler(epoch, lr):
    return lr if epoch<=5 else lr*np.exp(-0.1)
lr_scheduler = LearningRateScheduler(scheduler)

def train_and_evaluate_model(model=None, 
                            trainset='X4', 
                            **qwargs):

    data = sample_data(**qwargs)

    dictionary_of_trainsets = {
        'X4': [8, 9],
        'X0': [0, 1],
        'X123': [(slice(2,5)), slice(5,8)],
        'X6': [12, 13],    }
    y_train, y_test = data[-2], data[-1]

    X_train = data[dictionary_of_trainsets[trainset][0]]
    X_test = data[dictionary_of_trainsets[trainset][1]]

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adamax',
              metrics='accuracy')
    try:
        model.fit(X_train, y_train,
          validation_data = (X_test, y_test),
          batch_size=128,
          epochs=1,
          verbose=1,
          callbacks=[earlystopping],
          steps_per_epoch = 1,
          )
        return model.evaluate(X_test, y_test, verbose=2)[-1]

    except Exception as e:
        print(f"{model.name} failed to train.\n{str(e)}")
        return -1
