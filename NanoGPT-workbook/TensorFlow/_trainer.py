import tensorflow as tf
import logging

from _dataset_util import create_dataset_for_training
from _create_model_util import create_character_model





if __name__ ==  "__main__":

    URL = "./NanoGPT-workbook/data/tinyshakespeare.txt"
    BUFFER_SIZE = 10000
    VOCAB_SIZE = 66
    BLOCK_SIZE = 100
    BATCH_SIZE = 64
    DIMS = 32
    NUM_HEADS = 4
    FFNN_UNITS = 32
    DROPOUT_RATE = 0.2
    NUM_DECODER_LAYERS = 2
    NUM_EPOCHS = 5


    logging.basicConfig(level=logging.INFO)

    # Get the data
    dataset, mapper = create_dataset_for_training(URL, BLOCK_SIZE+1, BATCH_SIZE, BUFFER_SIZE)
    # Create training and validation dataset
    logging.info('[TRAIN-TEST-SPLIT]:Create train and test split')
    val_ds = dataset.take(22)
    train_ds = dataset.skip(22)

    # set the parameter
    VOCAB_SIZE = mapper['char_to_id'].vocabulary_size()

    # create the model
    model = create_character_model(
        VOCAB_SIZE, BLOCK_SIZE, NUM_DECODER_LAYERS, NUM_HEADS, DIMS, FFNN_UNITS, DROPOUT_RATE)
    
    # compile the model
    logging.info('[COMPILE THE MODEL]: Adam as optimizer, SparseCategorialCrossentropy as loss-function')
    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    # train the model   
    history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS)

    # SAVE MODEL

    






