

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

def train_and_evaluate_model(model, horizon_x4=1200):

    data = sample_data(target='WB 6',
                        horizon_x0=0,
                        horizon_x4=horizon_x4,
                        horizon_x5=0,
                        train_test_split=0.7,
                        scaling=True)
    X4_train = data[8]
    X4_test = data[9]
    y_train, y_test = data[-2], data[-1]
    print(X4_train.shape)

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adamax',
              metrics='accuracy')
    try:
        model.fit(X4_train, y_train,
          validation_data = (X4_test, y_test),
          batch_size=128,
          epochs=20,
          verbose=1,
          callbacks=[earlystopping],
          # steps_per_epoch = 1,
          )
        return model.evaluate(X4_test, y_test, verbose=2)[-1]

    except:
        print(f"{model.name} failed to train.")
        return -1
