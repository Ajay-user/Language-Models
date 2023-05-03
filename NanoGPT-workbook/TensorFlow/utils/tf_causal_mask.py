import tensorflow as tf
import logging
import time
import numpy as np



class Mask():
    def __init__(self, size):
        self.size = size
    
    def causal_mask(self,):
        ones = tf.ones(shape=[self.size, self.size])
        # triangular matrix ---> [size, size]
        lt = tf.linalg.band_part(ones, num_lower=-1, num_upper=0)
        return 1-lt
    


if __name__ == '__main__':


    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4

    logging.basicConfig(level=logging.INFO)

    start = time.time()

    sample_input = tf.random.normal(shape=[BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE])
    mask = Mask(size=BLOCK_SIZE).causal_mask()
    sample_output = sample_input + (mask*-1e9)

    end = time.time()

    print(f'TIME COST : {end-start}')


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of Attention output :', sample_output.shape)

    print('---- Display mask ----')

    mask = Mask(size=8).causal_mask()
    mask_output = np.where(mask == 1, -np.inf, 1)
    print(mask_output)
