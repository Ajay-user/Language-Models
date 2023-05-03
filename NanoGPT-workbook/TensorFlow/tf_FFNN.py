import tensorflow as tf
import logging
import time


class FFNN(tf.keras.layers.Layer):
    def __init__(self, ffnn_units, dims):
        super(FFNN, self).__init__()
        self.ffnn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffnn_units, activation='relu'),
            tf.keras.layers.Dense(dims)
        ])
    def call(self, inputs):
        return self.ffnn(inputs)
    




if __name__ == '__main__':


    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4
    FFNN_UNITS = 2048

    logging.basicConfig(level=logging.INFO)

    start = time.time()

    sample_input = tf.random.normal(shape=[BATCH_SIZE, BLOCK_SIZE, DIMS])

    ffnn = FFNN(ffnn_units=FFNN_UNITS, dims=DIMS)

    ffnn_output = ffnn(sample_input)

    end = time.time()

    print(f'TIME COST : {end-start}')


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of FFNN output :', ffnn_output.shape)

