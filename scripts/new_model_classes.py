import tensorflow as tf
from keras import Model
from keras.layers import Dense, Concatenate, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, Dropout, CategoryEncoding, Input
from keras.utils import set_random_seed, plot_model

class DensePlusOnePlusConvolutionModel(Model):
    def __init__(self, num_labels=50):
        super().__init__()
        # batch for input 1
        self.norm = BatchNormalization()
        # one hot encode for input 3
        self.onehotencode = CategoryEncoding(num_tokens=12, output_mode='one_hot')

        # concat for inputs 1-3
        self.concat1 = Concatenate()
        self.dense_date = Dense(32)

        # conv for input 4
        self.convs = [Conv2D(50, (4, 4), strides=1,
                            padding='same', activation='relu') for _ in range(3)]
        self.conv2s = [Conv2D(50, (4, 4), strides=1,
                            padding='same', activation='relu') for _ in range(3)]
        self.pools = [MaxPooling2D(3, padding='same') for _ in range(3)]
        self.flatten_x4 = Flatten()
        self.dense_x4 = Dense(32)
        
        
        # conv for input 5
        self.concat_conv = Concatenate(axis=2)
        self.conv1 = Conv2D(50, (1200, 2), strides=1, padding='valid', activation='relu')
        self.conv2 = Conv2D(50, (1, 2), strides=1, padding='valid', activation='relu')

        self.flatten_x5 = Flatten()
        self.conv_denses  = [Dense(256) for _ in range(3)]

        self.concat2 = Concatenate()
        self.denses = [Dense(unit) for unit in [256, 128, 128, 64]]


        # drop outs and output
        self.dropout1 = Dropout(0.05)
        self.dropout2 = Dropout(0.05)        
        self.output_layer = Dense(num_labels, activation='softmax')

    def call(self, inputs):
        x1, x2, x3, x4, x5 = inputs 
        x1 = self.norm(x1) # normalize day
        x3 = tf.cast(x3, 'int32') - 1
        x3 = self.onehotencode(x3) # onehot encode month

        # concat to x2
        x123 = self.concat1([x1, x2, x3])
        x123 = self.dense_date(x123)
        
        
        # convolution sub-model
        x4 = tf.cast(tf.expand_dims(x4, axis=-1), 'float')
        
        for conv, conv2, pool in zip(
            self.convs, self.conv2s, self.pools): # convolute x4 (windowed data)
            x4 = conv(x4)
            x4 = conv2(x4)
            x4 = pool(x4)
        x4 = self.flatten_x4(x4)
        x4 = self.dense_x4(x4)
        # x4 = self.dropout1(x4)


        # convolution sub-model
        x5 = tf.cast(tf.expand_dims(x5, axis=-1), 'float')

        x5 = self.concat_conv([x5, x5])
        x5 = self.conv1(x5)
        
        
        x5 = self.conv2(x5)
        x5 = self.flatten_x5(x5)
        
        for dense in self.conv_denses:
            x5 = dense(x5)
        # x5 = self.dropout1(x5)

        # concat the 4 inputs
        x = self.concat2([x123, x4, x5]) # then concat to the rest
        for dense in self.denses:
            x = dense(x)
        
        x = self.dropout2(x)
        return self.output_layer(x) 

    def build_graph(self, input_shape):
        """
        hack for visualizing nested layers
        """
        input1 = Input(shape=input_shape[0], name='Day')
        input2 = Input(shape=input_shape[1], name='Date_parameters')
        input3 = Input(shape=input_shape[2], name='Month')
        input4 = Input(shape=input_shape[3], name='Windowed_Time_series')
        input5 = Input(shape=input_shape[4], name='Custom_Windowed_Time_series')

        output = self.call([input1, input2, input3, input4, input5])
        model = Model(inputs=[input1, input2, input3, input4, input5], outputs=output)
        return model
    
            

    


class InputSubModelX123(Model):
    def __init__(self):
        super().__init__()
        self.norm = BatchNormalization()
        self.onehotencode = CategoryEncoding(num_tokens=12, output_mode='one_hot')
        self.concat1 = Concatenate()
        self.dense_date = Dense(32)

    def call(self, inputs):
        x1, x2, x3 = inputs
        x1 = self.norm(x1)  # normalize day
        x3 = tf.cast(x3, 'int32') - 1
        x3 = self.onehotencode(x3)  # onehot encode month
        x123 = self.concat1([x1, x2, x3])
        x123 = self.dense_date(x123)
        return x123


class ConvolutionSubModelX4(Model):
    def __init__(self):
        super().__init__()
        self.convs = [Conv2D(50, (4, 4), strides=1,
                             padding='same', activation='relu') for _ in range(3)]
        self.conv2s = [Conv2D(50, (4, 4), strides=1,
                              padding='same', activation='relu') for _ in range(3)]
        self.pools = [MaxPooling2D(3, padding='same') for _ in range(3)]
        self.flatten = Flatten()
        self.dense = Dense(32)

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
        return x4


class ConvolutionSubModelX5(Model):
    def __init__(self):
        super().__init__()
        self.concat_conv = Concatenate(axis=2)
        self.conv1 = Conv2D(50, (1200, 2), strides=1, padding='valid', activation='relu')
        self.conv2 = Conv2D(50, (1, 2), strides=1, padding='valid', activation='relu')
        self.flatten = Flatten()
        self.conv_denses = [Dense(256) for _ in range(3)]

    def call(self, inputs):
        x5 = inputs
        x5 = tf.cast(tf.expand_dims(x5, axis=-1), 'float')
        x5 = self.concat_conv([x5, x5])
        x5 = self.conv1(x5)
        x5 = self.conv2(x5)
        x5 = self.flatten(x5)
        for dense in self.conv_denses:
            x5 = dense(x5)
        return x5
    def build_graph(self, input_shape):
        input = Input(shape=input_shape)

        output = self.call(input)
        model = Model(inputs=[input], outputs=output)
        return model


class DensePlusOnePlusConvolutionModel(Model):
    def __init__(self, num_labels=50):
        super().__init__()
        self.submodel_x123 = InputSubModelX123()
        self.submodel_x4 = ConvolutionSubModelX4()
        self.submodel_x5 = ConvolutionSubModelX5()
        self.concat = Concatenate()
        self.denses = [Dense(unit) for unit in [256, 128, 128, 64]]
        self.dropout = Dropout(0.05)
        self.output_layer = Dense(num_labels, activation='softmax')

    def call(self, inputs):
        x1, x2, x3, x4, x5 = inputs
        x123 = self.submodel_x123([x1, x2, x3])
        x4 = self.submodel_x4(x4)
        x5 = self.submodel_x5(x5)
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


def test_DensePlusOnePlusConvolutionModel(horizon=1200):
    from IPython.core.display_functions import display
    model = DensePlusOnePlusConvolutionModel()

    # Call model.build() with the input shapes
    input_shape_x1 = (1,)  # day which is batched
    input_shape_x2 = (3,)  # the other day related parameters
    input_shape_x3 = (1,)  # month
    input_shape_x4 = (horizon, 6)  # horizon_convolution window of all 6 numbers
    input_shape_x5 = (horizon, 3)  # horizon_convolution window of 3 special numbers
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
                       show_shapes=True,))


test_DensePlusOnePlusConvolutionModel()
