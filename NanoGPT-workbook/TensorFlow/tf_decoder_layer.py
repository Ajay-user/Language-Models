import tensorflow as tf
import logging
import time

from tf_multi_head_attention import MultiHeadAttention
from tf_FFNN import FFNN


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, dims, ffnn_units, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads, dims)
        self.ffnn = FFNN(ffnn_units, dims)

        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, mask=None, training=False):
        # pre normalize
        vector = self.layer_norm_1(inputs)
        # attention
        attention = self.mha(vector, mask)
        # dropout
        out_1 = self.dropout_1(attention)
        # skip-connection
        out_1 += inputs

        # pre normalize
        vector = self.layer_norm_2(out_1)
        # Feed Forward Neural Nets
        ffnn_out = self.ffnn(vector)
        # dropout
        out_2 = self.dropout_2(ffnn_out)
        # skip-connection
        out_2 += out_1

        return out_2






if __name__ == '__main__':


    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4
    FFNN_UNITS = 2048
    DROPOUT_RATE = 0.2

    logging.basicConfig(level=logging.INFO)


    start = time.time()

    sample_input = tf.random.normal(shape=[BATCH_SIZE, BLOCK_SIZE, DIMS])

    decoder_layer = DecoderLayer(NUM_HEADS, DIMS, FFNN_UNITS, DROPOUT_RATE)

    sample_output = decoder_layer(sample_input)

    end = time.time()

    print(f'TIME COST : {end-start}')


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of Decoder layer output :', sample_output.shape)


    



        






