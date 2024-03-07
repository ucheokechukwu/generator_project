
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Concatenate, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, Dropout, CategoryEncoding, Input
from keras.utils import set_random_seed, plot_model



class InputSubModelX123(Model):
    def __init__(self):
        super().__init__()
        self.norm = BatchNormalization()
        self.onehotencode = CategoryEncoding(num_tokens=12, output_mode='one_hot')
        self.concat1 = Concatenate()
        self.dense_date = Dense(32)
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x1, x2, x3 = inputs
        x1 = self.norm(x1)  # normalize day
        x3 = tf.cast(x3, 'int32') - 1
        x3 = self.onehotencode(x3)  # onehot encode month
        x123 = self.concat1([x1, x2, x3])
        x123 = self.dense_date(x123)
        x123 = self.output_layer(x123)
        return x123


class ConvolutionSubModelX4(Model):
    def __init__(self, horizon=1200):
        super().__init__()
        self.convs = [Conv2D(44, (4, 4), strides=1,
                             padding='same', activation='relu') for _ in range(3)]
        self.conv2s = [Conv2D(44, (4, 4), strides=1,
                              padding='same', activation='relu') for _ in range(3)]
        self.pools = [MaxPooling2D(3, padding='same') for _ in range(3)]
        self.flatten = Flatten()
        self.dense = Dense(32)
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x4 = inputs
        x4 = tf.cast(tf.expand_dims(x4, axis=-1), 'float')
        for conv, conv2, pool in zip(
                self.convs, self.conv2s, self.pools):  # convolute x4 (windowed data)
            x4 = conv(x4)
            x4 = conv2(x4)
            x4 = pool(x4)
        x4 = self.flatten(x4)
        x4 = self.dense(x4)
        x4 = self.output_layer(x4)
        return x4


class ConvolutionSubModelX5(Model):
    def __init__(self, horizon=3000):
        super().__init__()
        self.concat_conv = Concatenate(axis=2)
        self.conv1 = Conv2D(44, (horizon, 2), strides=1, padding='valid', activation='relu')
        self.conv2 = Conv2D(44, (1, 2), strides=1, padding='valid', activation='relu')
        self.flatten = Flatten()
        self.conv_denses = [Dense(256) for _ in range(3)]
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x5 = inputs
        x5 = tf.cast(tf.expand_dims(x5, axis=-1), 'float')
        x5 = self.concat_conv([x5, x5])
        x5 = self.conv1(x5)
        x5 = self.conv2(x5)
        x5 = self.flatten(x5)
        for dense in self.conv_denses:
            x5 = dense(x5)
        
        x5 = self.output_layer(x5)
        return x5
    def build_graph(self, input_shape):
        input = Input(shape=input_shape)

        output = self.call(input)
        model = Model(inputs=[input], outputs=output)
        return model


class DensePlusOnePlusConvolutionModel(Model):
    def __init__(self):
        super().__init__()
        self.submodels = [InputSubModelX123(), 
                     ConvolutionSubModelX4(), 
                     ConvolutionSubModelX5(horizon=3000)
                    ]
        weights = ['submodel_weights/InputSubModelX123/3/ckpt/checkpoint.weights.h5',
                   'submodel_weights/ConvolutionSubModelX4/3/ckpt/checkpoint.weights.h5',
                   'submodel_weights/ConvolutionSubModelX5/3/ckpt/checkpoint.weights.h5'
                  ]

        input_shapes = [[(None, 1), (None, 3), (None, 1)], 
                        (None, 1200, 6), 
                        (None, 3000, 3)
                       ]
        for model, weight, input_shape in zip(self.submodels, weights, input_shapes):
            model.build(input_shape)
            model.load_weights(weight)
            model.trainable = False
        
        self.concat = Concatenate()
        self.denses = [Dense(unit) for unit in [256, 128, 128, 64]]
        self.dropout = Dropout(0.05)
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x1, x2, x3, x4, x5 = inputs
        x123 = self.submodels[0]([x1, x2, x3])
        x4 = self.submodels[1](x4)
        x5 = self.submodels[2](x5)
        x = self.concat([x123, x4, x5])
        for dense in self.denses:
            x = dense(x)
        x = self.dropout(x)
        return self.output_layer(x)

    def build_graph(self, input_shape):
        input1 = Input(shape=input_shape[0], name='Day')
        input2 = Input(shape=input_shape[1], name='Date_parameters')
        input3 = Input(shape=input_shape[2], name='Month')
        input4 = Input(shape=input_shape[3], name='Windowed_Time_series')
        input5 = Input(shape=input_shape[4], name='Custom_Windowed_Time_series')

        output = self.call([input1, input2, input3, input4, input5])
        model = Model(inputs=[input1, input2, input3, input4, input5], outputs=output)
        return model


def test_DensePlusOnePlusConvolutionModel(horizon_x4=1200, horizon_x5=3000):
    from IPython.core.display_functions import display
    model = DensePlusOnePlusConvolutionModel()

    # Call model.build() with the input shapes
    input_shape_x1 = (1,)  # day which is batched
    input_shape_x2 = (3,)  # the other day related parameters
    input_shape_x3 = (1,)  # month
    input_shape_x4 = (horizon_x4, 6)  # horizon_convolution window of all 6 numbers
    input_shape_x5 = (horizon_x5, 3)  # horizon_convolution window of 3 special numbers
    model.build(input_shape=[(None,) + input_shape_x1,
                             (None,) + input_shape_x2,
                             (None,) + input_shape_x3,
                             (None,) + input_shape_x4,
                             (None,) + input_shape_x5
                             ])
    display(plot_model(model.build_graph([input_shape_x1,
                                          input_shape_x2,
                                          input_shape_x3,
                                          input_shape_x4,
                                          input_shape_x5]),
                       expand_nested=True,
                       show_shapes=True,
                       show_trainable=True))
    return model

    
if __name__ == "__main__":
    test_DensePlusOnePlusConvolutionModel()
