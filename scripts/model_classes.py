import tensorflow as tf
from keras.layers import Input, Dropout, Conv1D, Conv2D, Flatten, Dense
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Concatenate
from keras import Model
from keras.utils import plot_model, set_random_seed
set_random_seed(0)


class ConvolutionalModel(Model):
    def __init__(self):
        super().__init__()
        self.conv = [Conv2D(49, (6, 6), strides=1,
                            padding='same', activation='relu') for _ in range(12)]
        self.flatten = Flatten()
        self.dense1 = Dense(8, activation='relu')
        self.dense2 = Dense(8, activation='relu')
        self.output_layer = Dense(49, activation="softmax")

    def call(self, inputs):
        x = inputs
        for conv in self.conv:
            x = conv(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.multi_head_attention = MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)
        self.dropout1 = Dropout(self.dropout)
        self.conv1 = Conv1D(filters=self.ff_dim,
                            kernel_size=1, activation="relu")
        self.dropout2 = Dropout(self.dropout)
        self.conv2 = Conv1D(filters=input_shape[-1], kernel_size=1)
        super().build(input_shape)

    def call(self, inputs):
        x = self.multi_head_attention(inputs, inputs)
        x = self.dropout1(x)
        res = x + inputs
        x = self.conv1(res)
        x = self.dropout2(x)
        x = self.conv2(x)
        return x + res


class AttentionModel(tf.keras.Model):
    def __init__(self, head_size=256, num_heads=4, ff_dim=1, num_transformer_blocks=4, mlp_units=[128], dropout=0.0, mlp_dropout=0.0):
        super().__init__()

        self.transformer_blocks = [TransformerEncoder(
            head_size, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)]
        self.global_average_pooling = GlobalAveragePooling1D(
            data_format="channels_last")
        self.dense_layers = [Dense(dim, activation="relu")
                             for dim in mlp_units]
        self.dropout_layer = Dropout(mlp_dropout)
        self.output_layer = Dense(50, activation="softmax")

    def call(self, inputs):
        x = inputs
        for block in self.transformer_blocks:
            x = block(x)
        x = self.global_average_pooling(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.dropout_layer(x)
        return self.output_layer(x)


class ConvolutionalModel(Model):
    def __init__(self):
        super().__init__()
        self.convs = [Conv2D(49, (6, 6), strides=1,
                             padding='same', activation='relu') for _ in range(12)]
        # self.conv1 = Conv2D(49, (3, 3), strides=1,
        #                     padding='same', activation='relu')
        # self.conv2 = Conv2D(49, (3, 3), strides=1,
        #                     padding='same', activation='relu')
        # self.conv3 = Conv2D(49, (3, 3), strides=1,
        #                     padding='same', activation='relu')
        # self.conv4 = Conv2D(49, (3, 3), strides=1,
        #                     padding='same', activation='relu')
        # self.conv5 = Conv2D(49, (3, 3), strides=1,
        #                     padding='same', activation='relu')
        # self.conv6 = Conv2D(49, (3, 3), strides=1,
        #                     padding='same', activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(8, activation='relu')
        self.dense2 = Dense(8, activation='relu')
        self.output_layer = Dense(49, activation="softmax")

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output


class ConvolutionalModelwithoutPadding(Model):
    def __init__(self):
        super().__init__()
        self.convs = [Conv2D(49, (3, 3), strides=1,
                             padding='valid', activation='relu') for _ in range(3)]

        self.flatten = Flatten()
        self.dense1 = Dense(8, activation='relu')
        self.dense2 = Dense(8, activation='relu')
        self.output_layer = Dense(49, activation="softmax")

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
            print(x.shape)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output
