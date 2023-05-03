import tensorflow as tf
import logging
import time

from tf_decoder_layer import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, dims, ffnn_units, dropout_rate):
        super(Decoder, self).__init__()
        self.decoder_layers = [
            DecoderLayer(num_heads, dims, ffnn_units, dropout_rate) for _ in range(num_layers)
        ]

    def call(self, vectors, mask=None, training=False):
        
        for block in self.decoder_layers:
            vectors = block(vectors, mask, training)

        return vectors
    




if __name__ == '__main__':


    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4
    FFNN_UNITS = 2048
    DROPOUT_RATE = 0.2
    NUM_DECODER_LAYERS = 4

    logging.basicConfig(level=logging.DEBUG)


    start = time.time()

    sample_input = tf.random.normal(shape=[BATCH_SIZE, BLOCK_SIZE, DIMS])

    decoder = Decoder(NUM_DECODER_LAYERS,NUM_HEADS, DIMS, FFNN_UNITS, DROPOUT_RATE)

    sample_output = decoder(sample_input)

    end = time.time()

    print(f'TIME COST : {end-start}')


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of Decoder output :', sample_output.shape)

