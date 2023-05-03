import tensorflow as tf
import numpy as np
import pathlib
import logging

from _dataset_util import create_dataset_for_training

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)


    URL = "./NanoGPT-workbook/data/tinyshakespeare.txt"
    BUFFER_SIZE = 10000
    VOCAB_SIZE = 66
    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 512
    NUM_HEADS = 4
    FFNN_UNITS = 2048
    DROPOUT_RATE = 0.2
    NUM_DECODER_LAYERS = 4


    # getting data
    dataset, mapper = create_dataset_for_training(
        url=URL, block_size=BLOCK_SIZE+1, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)


    message = f'''
    ✅ Character Model Configuration 🤖🔧🔧\n
    :::::::::::::::::::::::\n
    📖VOCAB_SIZE = {VOCAB_SIZE}\n
    🚥BLOCK_SIZE = {BLOCK_SIZE}\n
    🚥BATCH_SIZE = {BATCH_SIZE}\n
    🚥DIMS = {DIMS}\n
    🔨NUM_HEADS = {NUM_HEADS}\n
    🔩FFNN_UNITS = {FFNN_UNITS}\n
    🪂DROPOUT_RATE = {DROPOUT_RATE}\n
    🔧NUM_DECODER_LAYERS = {NUM_DECODER_LAYERS}\n
    :::::::::::::::::::::::\n
    '''
    
    logging.info(message)


    logging.info('[TASK] >>>>> Loading Model')
    model = tf.keras.models.load_model(filepath='./NanoGPT-workbook/TensorFlow/saved_model')


    logging.info('Model Evaluation 🧾')
    model.evaluate(dataset.take(10))


    
    

 