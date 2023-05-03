import tensorflow as tf
import time
import logging

from data_processing.tf_data import get_data
from data_processing.tf_vocab_mappers import get_character_maps
from data_processing.tf_create_dataset import create_sequence_blocks, create_dataset


def create_dataset_for_training(
        url:str, block_size:int,
        batch_size:int, buffer_size:int)->tuple[tf.data.Dataset, dict[str, tf.keras.layers.Layer]]:
    '''
    Create a dataset for training\n
    Step 1 : Download or read the data\n
    Step 2 : Create mappers [char to integer-ids and viceversa]\n
    Step 3 : Create Fixed Length blocks\n
    Step 4 : Create a Tensorflow Dataset\n

    return the dataset and mapper
    '''
    logging.info('[TASK] >>>>> Create Training Dataset 💾')
    # download the data
    data = get_data(url)
    # create maps eg: {char_to_id, id_to_char, id_to_str} 
    mapper = get_character_maps(vocabulary=data)
    # create fixed length sequences
    ds = create_sequence_blocks(data, block_size, mapper)
    # create final dataset
    dataset = create_dataset(ds, batch_size, buffer_size)
    logging.info('[JOB FINISHED] >>>>> Training Dataset created 💾 ✅')
    return dataset, mapper





if __name__ == "__main__":

    BLOCK_SIZE = 101
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    URL = "./NanoGPT-workbook/data/tinyshakespeare.txt"

    logging.basicConfig(level=logging.INFO)

    start = time.time()

    ds, _ = create_dataset_for_training(URL, BLOCK_SIZE, BATCH_SIZE, BUFFER_SIZE)
 
    end = time.time()

    print(f'TIME COST : {end-start}')

   

