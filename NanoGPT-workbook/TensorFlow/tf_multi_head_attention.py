import tensorflow as tf
import logging
import time

from tf_self_attention import Attention


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads, dims):
        super(MultiHeadAttention, self).__init__()
        self.dims = dims
        self.heads = num_heads
        self.head_dims = dims//num_heads
        tf.assert_equal(
            x=dims%num_heads, y=0, message='Multi head attention condition : heads*head_dims = embed_dims')


        self.dq = tf.keras.layers.Dense(dims)
        self.dk = tf.keras.layers.Dense(dims)
        self.dv = tf.keras.layers.Dense(dims)

        self.dense = tf.keras.layers.Dense(dims)

        self.attention = Attention()

    def split_heads(self, vector):
        # [batch, seq, dims] --> [batch, seq, head, head-dims]
        vector = tf.reshape(vector, shape=[self.batch_size, -1, self.heads, self.head_dims])
        # [batch, seq, head, head-dims] --> [batch, head, seq, head-dims]
        vector = tf.transpose(vector, perm=[0,2,1,3])
        return vector
    
    def concat_heads(self, vector):
        # [batch, head, seq, head-dims] --> [batch, seq, head, head-dims]
        vector = tf.transpose(vector, perm=[0,2,1,3])
        # [batch, seq, head, head-dims] --> [batch, seq, dims]
        vector = tf.reshape(vector, shape=[self.batch_size, -1, self.heads*self.head_dims])
        return vector
    
    def call(self, inputs, mask=None):

        self.batch_size = tf.shape(inputs)[0]
        # keys, queries and values  |  shape :[batch, seq]
        k, q, v = [inputs]*3

        # projection --> [batch, seq, dims]
        q = self.dq(q)
        k = self.dk(k)
        v = self.dv(v)

        # split heads --> [batch, head, seq, head-dims]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # attention --> [batch, head, seq, head-dims]
        self_attention = self.attention(k, q, v, mask=mask)

        # concat | [batch, head, seq, head-dims] --> [batch, seq, dims]
        vector = self.concat_heads(self_attention)

        # project --> [batch, seq, dims]
        vector = self.dense(vector)

        return vector






if __name__ == '__main__':


    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4

    logging.basicConfig(level=logging.INFO)

    start = time.time()

    sample_input = tf.random.normal(shape=[BATCH_SIZE, BLOCK_SIZE, DIMS])

    mha = MultiHeadAttention(num_heads=NUM_HEADS, dims=DIMS)
    multi_head_attention_output = mha(sample_input)


    end = time.time()

    print(f'TIME COST : {end-start}')


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of Attention output :', multi_head_attention_output.shape)


    



        






