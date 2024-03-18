from tensorflow.keras.layers import Layer
import tensorflow as tf

class CustomNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # Calculate min, max, mean, and scale
#         tf.print(inputs)
        min_val = tf.reduce_min(inputs, axis=0)
        max_val = tf.reduce_max(inputs, axis=0)
        mean = tf.reduce_mean(inputs, axis=0)
        scale = tf.constant(1.0) / (max_val - min_val)

        # Normalize and scale the inputs to [-0.5, 0.5]
        normalized_scaled = scale * (inputs - mean)
#         tf.print(normalized_scaled)

        return normalized_scaled

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
from keras.layers import Input, CategoryEncoding, Concatenate, Dense, BatchNormalization
from keras import Model
class InputSubModelX123(Model):
    def __init__(self, norm=CustomNormalization()):
        super().__init__()
        self.norm = norm
        self.onehotencode = CategoryEncoding(num_tokens=12, output_mode='one_hot')
        self.concat1 = Concatenate()
        self.dense_date = Dense(32)
        self.output_layer = Dense(44, activation='softmax')

    def call(self, inputs):
        x1, x2, x3 = inputs
        x1 = self.norm(x1)  # normalize day
        
#         tf.print(x1)
        x3 = tf.cast(x3, 'int32') - 1
        x3 = self.onehotencode(x3)  # onehot encode month
        x123 = self.concat1([x1, x2, x3])
        x123 = self.dense_date(x123)
        x123 = self.output_layer(x123)
        return x123
    
    
    def build_graph(self, input_shape):
        x1, x2, x3 = input_shape
        in_ = [Input(shape=x1), Input(shape=x2), Input(shape=x3)]
        out_ = self.call(in_)
        return Model(inputs=in_, outputs=out_)