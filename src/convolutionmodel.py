
import tensorflow as tf
from keras.layers import Input, Dropout, Conv1D, Conv2D, Flatten, Dense, MaxPooling2D
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Concatenate
from keras import Model
from keras.utils import plot_model, set_random_seed
set_random_seed(0)

class ConvolutionBlock(Model):
    def __init__(self, kernel, pooling):
        
        super().__init__()
        self.convs = [Conv2D(44, kernel, strides=1,
                             padding='same', activation='relu') for _ in range(3)]
        self.conv2s = [Conv2D(44, kernel, strides=1,
                              padding='same', activation='relu') for _ in range(3)]

        self.pools = [MaxPooling2D(kernel[1], padding='same') for _ in range(3)]
        self.pooling = pooling


    def call(self, inputs):
        x4 = inputs
        for conv, conv2, pool in zip(
                self.convs, self.conv2s, self.pools):  # convolute x4 (windowed data)
            x4 = conv(x4)
            x4 = conv2(x4)
            x4 = pool(x4) if self.pooling else x4
        return x4
    
    
    
class ConvolutionModel(tf.keras.Model):
    def __init__(self, 
                 num_convolution_blocks = 1,
                 num_dense_layers = 1,
                 kernel = (4,4),
                 pooling = True):
        super().__init__()

        self.convolution_blocks = [ConvolutionBlock(
                                            kernel=kernel,
                                            pooling=pooling,
                                            ) for _ in range(num_convolution_blocks)]
        self.flatten = Flatten()
        self.denses = [Dense(64) for _ in range(num_dense_layers)]
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = tf.cast(tf.expand_dims(x, axis=-1), 'float')
#         print(x.shape)
        for block in self.convolution_blocks:
            x = block(x)
#             print(x.shape)
        x = self.flatten(x)
#         print(x.shape)
        for dense in self.denses:
            x = dense(x)
#             print(x.shape)
        x = self.output_layer(x)
#         print(x.shape)
        return x
    
    def build_graph(self, horizon=1200):
        input_x = Input(shape=(horizon, 6))
        output = self.call(input_x)
        model = Model(inputs=input_x, outputs=output)
        return model
    
m = ConvolutionModel(num_convolution_blocks=1,
                    num_dense_layers=3,
                    kernel=(1000,6),
                    pooling=True).build_graph(horizon=3000)
plot_model(m, show_shapes=True)
