
import tensorflow as tf
from keras.layers import (
    Input, 
    Dropout, 
    Conv1D, 
    Conv2D, 
    Flatten, 
    Dense, 
    Layer, 
    Concatenate, 
    MaxPooling2D,
    Normalization, 
    CategoryEncoding
    )
from keras import Model
from keras.utils import plot_model, set_random_seed
set_random_seed(0)




# Timeseries
import tensorflow as tf
from keras.layers import (
    Input, 
    Dropout, 
    Add,
    Conv1D, 
    Conv2D, 
    Flatten, 
    Dense, 
    Layer, 
    Concatenate, 
    MultiHeadAttention, 
    GlobalAveragePooling1D, 
    GlobalMaxPooling1D
    )
from keras import Model
from keras.utils import plot_model, set_random_seed
set_random_seed(0)



class SubModelX0(Model):
    """Time series model to work with data X0"""
    def __init__(self, 
                 mlp_units=[128, 44, 44], 
                 activation='relu',
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dense_layers = [Dense(dim, activation=activation) for dim in mlp_units]
        self.output_layer = Dense(44, activation="softmax")

    def call(self, inputs):
        
        inputs = tf.squeeze(inputs, axis=-1)
        x = inputs
        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return self.output_layer(x)
    
    def build_graph(self, input_shape):
        in_ = Input(shape=input_shape)
        out_ = self.call(in_)
        model = Model(inputs = in_, outputs = out_)
        return model

    
class SubModelX123(Model):
    def __init__(self, 
                 mlp_units=[128], 
                 activation='relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.norm = Normalization()
        self.onehotencode = CategoryEncoding(num_tokens=12, output_mode='one_hot')
        self.concat1 = Concatenate()
        self.dense_layers = [Dense(dim, activation=activation) for dim in mlp_units
                            ] if mlp_units else []
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x1, x2, x3 = inputs
        x1 = self.norm(x1)  # normalize day
        x3 = tf.cast(x3, 'int32') - 1
        x3 = self.onehotencode(x3)  # onehot encode month
        x123 = self.concat1([x1, x2, x3])
        for dense_layer in self.dense_layers:
            x123 = dense_layer(x123)
            
        x123 = self.output_layer(x123)
        return x123
    
class ConvolutionSubModelX4(Model):
    def __init__(self, 
                 horizon=1200, 
                 num_blocks=1, 
                 mlp_units=[64],
                 **kwargs):

        super().__init__(**kwargs)
        self.convs = [Conv2D(44, (horizon, 2), strides=1,
                             padding='valid', activation='relu') for _ in range(num_blocks)]
        self.conv2s = [Conv2D(44, (4, 4), strides=1,
                              padding='valid', activation='relu') for _ in range(num_blocks)]
        self.pools = [MaxPooling2D(3, padding='valid') for _ in range(num_blocks)]
        self.flatten = Flatten()
        
        self.dense_layers = [Dense(dim, activation='relu') for dim in mlp_units
                            ] if mlp_units else []
        
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x4 = inputs
        x4 = tf.cast(tf.expand_dims(x4, axis=-1), 'float')
        for conv, conv2, pool in zip(
                self.convs, self.conv2s, self.pools):  # convolute x4 (windowed data)
            x4 = conv(x4)
#             x4 = conv2(x4)
#             x4 = pool(x4)
        x4 = self.flatten(x4)
        for dense_layer in self.dense_layers:
            x4 = dense_layer(x4) if self.dense_layers else x4
        x4 = self.output_layer(x4)
        return x4
    
    
