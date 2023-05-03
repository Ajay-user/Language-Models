import tensorflow as tf
import logging
import time




class Attention(tf.keras.layers.Layer):
    def __init__(self, ):
        super(Attention, self).__init__()
    
    def call(self, k, q, v, mask=None):
        # dimension of key
        k_dims = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    
        # [batch, query, dims] @ [batch, key, dims]^T = [batch, query, keys]
        product = tf.matmul(q, k, transpose_b=True)

        # scaling
        prod_scaled = product / tf.math.sqrt(k_dims)

        # apply mask
        if mask is not None:
            prod_scaled += (mask*-1e9)
        
        # softmax
        weights = tf.nn.softmax(prod_scaled, axis=-1)

        # attention [batch, query, keys] @ [batch, value, dims] --> [batch, value, dims]
        attention = tf.matmul(weights, v)

        return attention



if __name__ == '__main__':


    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512


    logging.basicConfig(level=logging.INFO)

    start = time.time()

    sample_input = tf.random.normal(shape=[BATCH_SIZE, BLOCK_SIZE, DIMS])
    k, q, v = [sample_input]*3

    attention = Attention()
    attention_output = attention(k, q, v)


    end = time.time()

    print(f'TIME COST : {end-start}')

    print('Shape of query Embedding :',q.shape)
    print('Shape of query Embedding :',k.shape)
    print('Shape of query Embedding :',v.shape)
    print('Shape of Attention output :',attention_output.shape)


    

