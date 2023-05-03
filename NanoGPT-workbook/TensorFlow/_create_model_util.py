import tensorflow as tf
import numpy as np
import logging
import time

from tf_character_model import CharacterModel


def create_character_model(
        vocab_size:int, 
        positions:int, 
        num_layers:int, 
        num_heads:int, 
        dims:int, 
        ffnn_units:int, 
        dropout_rate:float)->tf.keras.Model:
    '''
    Creates a character model using tensorflow
    '''
    logging.info('[TASK] >>>>> Create Tensorflow character model 🤖')
    model = CharacterModel(vocab_size, positions, num_layers, num_heads, dims, ffnn_units, dropout_rate)
    logging.info('[JOB FINISHED] >>>>> Tensorflow character model 🤖 ✅')
    return model





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

    model = create_character_model(
        VOCAB_SIZE, BLOCK_SIZE, NUM_DECODER_LAYERS, NUM_HEADS, DIMS, FFNN_UNITS, DROPOUT_RATE)

    sample_output = model(sample_input)

    end = time.time()

    print(f'TIME COST : {end-start}')


    print('Shape of input Embedding :', sample_input.shape)
    print('Shape of Decoder output :', sample_output.shape)

