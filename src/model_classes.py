import tensorflow as tf
from keras.layers import Input, Dropout, Conv1D, Conv2D, Flatten, Dense, Layer
from keras.layers import LayerNormalization
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Concatenate
from keras import Model
from keras.utils import plot_model, set_random_seed
set_random_seed(0)


class TransformerEncoder(Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.multi_head_attention = MultiHeadAttention(
            key_dim=self.head_size, 
            num_heads=self.num_heads, 
            dropout=self.dropout)
        self.dropout1 = Dropout(self.dropout)
        self.normalization1 = LayerNormalization(epsilon=1e-6)
        self.conv1 = Conv1D(filters=self.ff_dim,
                            kernel_size=1, activation="relu")
        self.dropout2 = Dropout(self.dropout)
        self.conv2 = Conv1D(filters=input_shape[-1], kernel_size=1)
        self.normalization2 = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, inputs):
        x = self.multi_head_attention(inputs, inputs)
        x = self.dropout1(x)
        x = self.normalization1(x) # optional normalization
        res = x + inputs
        x = self.conv1(res)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.normalization2(x) # optional normalization
        return x + res


class AttentionModel(Model):
    def __init__(self, 
                 head_size=256, 
                 num_heads=4, 
                 ff_dim=4, 
                 num_transformer_blocks=4, 
                 mlp_units=[128], 
                 dropout=0.25, 
                 mlp_dropout=0.4):
        super().__init__()
        self.transformer_blocks = [TransformerEncoder(
            head_size, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)]
        self.global_average_pooling = GlobalAveragePooling1D(
            data_format="channels_last")
        self.dense_layers = [Dense(dim, activation="relu") for dim in mlp_units]
        self.dropout_layers = [Dropout(mlp_dropout) for dim in mlp_units]
        self.output_layer = Dense(44, activation="softmax")

    def call(self, inputs):
        x = inputs
        for block in self.transformer_blocks:
            x = block(x)
        x = self.global_average_pooling(x)
        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            x = dense_layer(x)
            x = dropout_layer(x)

        return self.output_layer(x)
    
    def build_graph(self, input_shape):
        in_ = Input(shape=input_shape)
        out_ = self.call(in_)
        model = Model(inputs = in_, outputs = out_)
        return model




