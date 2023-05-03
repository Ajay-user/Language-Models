import tensorflow as tf
import numpy as np
import logging
import time


from tf_decoder import Decoder

from utils.tf_causal_mask import Mask
from utils.tf_positional_embedding import PositionalEmbedding


class CharacterModel(tf.keras.Model):
    def __init__(self, vocab_size, positions, num_layers, num_heads, dims, ffnn_units, dropout_rate):
        super(CharacterModel, self).__init__()
        self.dims = dims

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=dims)
        self.PE = PositionalEmbedding(positions, dims).get_positional_embeddings()
        self.mask = Mask(size=positions).causal_mask()
        self.decoder = Decoder(num_layers, num_heads, dims, ffnn_units, dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # shape of input [batch, sequences]
        (batch, sequences) = inputs.shape
        # prepare causal mask
        causal_mask = self.mask[:sequences, :sequences]
        # prepare positional embeddings
        if training:
            pe = self.PE
        else:
            pe = self.PE[:,:sequences,:]

        # embeddings --> [batch, sequences, dims]
        embed = self.embedding(inputs)

        # scale the embeddings
        factor = tf.math.sqrt(tf.cast(self.dims, dtype=tf.float32))
        embed *= factor

        # Meaning and Word order --> [batch, sequences, dims]
        vector = embed + pe

        # Context, Meaning and Word order --> [batch, sequences, dims]
        vector = self.decoder(vector, mask=causal_mask, training=training)
        
        # layer norm
        vector = self.layer_norm(vector)

        # dropout layer
        vector = self.dropout(vector)

        # output layer
        logits = self.dense(vector)

        return logits
    
    @tf.function
    def train_step(self, inputs):
        X, y = inputs
        with tf.GradientTape() as tape:
            logits = self(X, training=True)
            loss = self.compiled_loss(y_true=y, y_pred=logits)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
        self.compiled_metrics.update_state(y_true=y, y_pred=logits)
        return {m.name: m.result() for m in self.metrics}

    




if __name__ == '__main__':

    VOCAB_SIZE = 66
    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4
    FFNN_UNITS = 2048
    DROPOUT_RATE = 0.2
    NUM_DECODER_LAYERS = 4


    logging.basicConfig(level=logging.DEBUG)


    start = time.time()

    sample_input = np.random.randint(low=0, high=VOCAB_SIZE ,size=[BATCH_SIZE, BLOCK_SIZE])

    model = CharacterModel(
        VOCAB_SIZE, BLOCK_SIZE, NUM_DECODER_LAYERS, NUM_HEADS, DIMS, FFNN_UNITS, DROPOUT_RATE)

    sample_output = model(sample_input)

    end = time.time()

    print(f'TIME COST : {end-start}')


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of Decoder output :', sample_output.shape)











