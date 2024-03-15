
set_random_seed(0)
from keras import callbacks
from helper_files import *


def scheduler(epoch, lr):
    return lr if epoch<=5 else lr*np.exp(-0.1)
# lr_scheduler = callbacks.LearningRateScheduler(scheduler)
# earlystopping = callbacks.EarlyStopping(patience=50, restore_best_weights=True)
checkpoint_filepath = '/tmp/ckpt6/checkpoint.weights.h5'
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model = DensePlusOnePlusConvolutionModel()
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adamax',
              metrics=['accuracy'])

model.fit([X1_train, X2_train, X3_train, X4_train], y6_train, 
          shuffle=True,
          batch_size=32,
          epochs=3, 
          verbose=1, 
          validation_data=[[X1_test, X2_test, X3_test, X4_test], y6_test], 
          validation_batch_size=128,
          callbacks = [model_checkpoint_callback]
          )

model.load_weights(checkpoint_filepath)
!rm -r '/tmp/ckpt6/checkpoint.weights.h5'
results = model.evaluate([X1_test, X2_test, X3_test, X4_test], y6_test)

# plot_training(model)
