
# Functions from previous notebook

# import dependencies
from model_classes import *
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



def make_model(horizon=6, data_points=6):
    model = Sequential()
    # 5 by 4

    model.add(InputLayer(input_shape=(horizon, data_points, 1)))

    model.add(Conv2D(1, (3, 3), strides=1, activation='relu'))
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
    return tf.squeeze(forecast)  # return 1D array of predictions


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


def make_clf(horizon=6, num_balls=6):
    model = Sequential()
    # 5 by 4

    model.add(InputLayer(input_shape=(horizon, num_balls, 1)))

    # model.add(Conv2D(1, (3,3), strides=1, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dense(50, activation="softmax"))

    # model.build()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    # model.summary()
    return model


def make_attention_model(input_shape):
    attn_input_layer = Input(shape=input_shape)
    attn_layer = AttentionModel(
        # input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=1,
        num_transformer_blocks=4,
        mlp_units=[128],
        # mlp_dropout=0.4,
        # dropout=0.25,
    )
    attn_output = attn_layer(attn_input_layer)
    model = Model(inputs=attn_input_layer,
                  outputs=attn_output, name="attention")
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def create_attention_convolution_model(horizon_attention, horizon_convolution):

    # attention model
    attn_input_layer = Input(
        shape=(horizon_attention, 1), name="Attention_InputLayer")
    attn_layer = AttentionModel()
    attn_output = attn_layer(attn_input_layer)
    attn_model = Model(inputs=attn_input_layer,
                       outputs=attn_output, name="attention")

    # conv model
    conv_input_layer = Input(
        shape=(horizon_convolution, 8, 1), name="Convolution_InputLayer")
    conv_layer = ConvolutionalModel()
    conv_output = conv_layer(conv_input_layer)
    conv_model = Model(inputs=conv_input_layer,
                       outputs=conv_output, name="convolution")

    # Concatenate model outputs
    x = Concatenate()([attn_model.output, conv_model.output])

    # Create output layers
    x = Dense(32, activation="relu")(x)
    output_layer = Dense(50, activation="softmax")(x)

    # 5. Construct model with char and token inputs
    model = tf.keras.Model(inputs=[attn_model.input, conv_model.input],
                           outputs=output_layer,
                           name="combined_model")

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    plot_model(model, show_shapes=True)
    return model
