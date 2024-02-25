
## Functions from previous notebook

# import dependencies
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import InputLayer
from keras.layers import Dense


# convert this into a dataset of 6x6 with a window 

def transform_data(data, horizon=6):
    # Convert the DataFrame to a numpy array
    data_array = data.to_numpy()

    # Calculate the number of slices
    num_slices = len(data) - horizon + 1

    # Use numpy's array slicing to create the transformed data
    data_transformed = np.array([data_array[i:i+horizon] for i in range(num_slices)])
    return data_transformed[:-1], data_array[horizon:][:,1:-1]


# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels


def get_data(data, horizon):

    windows, labels = transform_data(data, horizon=horizon)
    windows = tf.expand_dims(windows, axis=-1)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
       windows=windows, 
       labels=labels, test_split=0.1)
    return train_windows, test_windows, train_labels, test_labels

# model builder

def make_model(horizon=6, data_points=6):
    model = Sequential()
    # 5 by 4

    model.add(InputLayer(input_shape=(horizon, data_points, 1)))

    model.add(Conv2D(1, (3,3), strides=1, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(6))

    # model.build()
    model.compile(loss="mae",
                        optimizer=tf.keras.optimizers.Adam())
    # model.summary()
    return model


def make_preds(model, input_data):
  """
  Uses model to make predictions on input_data.

  Parameters
  ----------
  model: trained model 
  input_data: windowed input data (same kind of data model was trained on)

  Returns model predictions on input_data.
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # return 1D array of predictions


# evaluation metrics
def evaluate_preds(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.int32)
  y_pred = tf.cast(y_pred, dtype=tf.int32)

  acc = []
  for i in range(len(y_true)):
    accuracy_i = accuracy_score(y_true[i], y_pred[i])
    acc.append(accuracy_i)
  acc = sum(acc)/len(acc)
  
  return {"accuracy": acc}