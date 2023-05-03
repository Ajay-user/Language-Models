import tensorflow as tf
import numpy as np
import logging
import time


class PositionalEmbedding():
    def __init__(self, positions, dims):
        self.positions = positions
        self.dims = dims
    
    def get_angle(self, pos, i, dims):
        deno = 10000**(2*(i//2) / dims )
        theta = np.divide(pos, deno)
        return theta
    
    def get_positional_embeddings(self,):
        positions = np.arange(self.positions)[:, np.newaxis]
        i = np.arange(self.dims)[np.newaxis, :]
        theta = self.get_angle(pos=positions, i=i, dims=self.dims)

        theta[:, 0::2] = np.sin(theta[:, 0::2])
        theta[:, 1::2] = np.cos(theta[:, 1::2])

        pe = tf.cast(theta[np.newaxis,:], dtype=tf.float32)
        return pe
    




if __name__ == '__main__':


    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4

    logging.basicConfig(level=logging.INFO)

    start = time.time()

    sample_input = tf.random.normal(shape=[BATCH_SIZE, BLOCK_SIZE, DIMS])
   
    PE = PositionalEmbedding(positions=BLOCK_SIZE, dims=DIMS)
    positional_embedding = PE.get_positional_embeddings()

    end = time.time()

    print(f'TIME COST : {end-start}')

    # import matplotlib.pyplot as plt
    # plt.imshow(positional_embedding[0])
    # plt.show()


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of Attention output :', positional_embedding.shape)


        
