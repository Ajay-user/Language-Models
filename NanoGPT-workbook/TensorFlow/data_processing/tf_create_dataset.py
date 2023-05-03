import tensorflow as tf
import time
import logging



def inputs_and_targets(text_ids:tf.Tensor)->tuple[tf.Tensor,tf.Tensor]:
    '''
    Split the sequence into inputs and target 
    '''
    inputs = text_ids[:-1]
    targets = text_ids[1:]
    return inputs, targets
    


def create_sequence_blocks(
        text:str, block_length:int, mapper:dict[str, tf.keras.layers.Layer])->tf.data.Dataset:
    '''
    Creates input & target sequences of fixed length \n

    step 1 : input string is mapped into integer ids \n
    step 2 : create fixed length sequences
    step 3 : create inputs and tragets
    '''

    logging.info('Mapping: map string to integer ids...')
    all_ids = mapper['char_to_id'](tf.strings.unicode_split(text, input_encoding='UTF-8'))

    logging.info('Creating fixed length sequences...')
    ds = (tf.data.Dataset.from_tensor_slices(all_ids)
          .batch(block_length, drop_remainder=True)
          .map(inputs_and_targets))
    logging.info(f'Fixed length Sequence created : block size = {block_length} ')
    return ds


def create_dataset(
        ds:tf.data.Dataset, batch_size:int, buffer_size:int)->tf.data.Dataset:
    '''
    Create dataset : we can feed this to a tensorflow model in fit() method
    '''
    logging.info(f'Creating-Datset: batch size = {batch_size}')
    return ds.batch(batch_size, drop_remainder=True).shuffle(buffer_size)




if __name__ == "__main__":


    from tf_data import get_data
    from tf_vocab_mappers import get_character_maps

    BLOCK_SIZE = 101
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000


    logging.basicConfig(level=logging.INFO)

    start = time.time()

    data = get_data("./NanoGPT-workbook/data/tinyshakespeare.txt")
    mapper = get_character_maps(vocabulary=data)
    ds = create_sequence_blocks(data, BLOCK_SIZE, mapper)
    dataset = create_dataset(ds, BATCH_SIZE, BUFFER_SIZE)
    
    end = time.time()

    print(f'TIME COST : {end-start}')
    print('Vocab size :',mapper['char_to_id'].vocabulary_size())
    print('Sequence Specs :',ds.element_spec)
    print('Dataset Specs:', dataset.element_spec)

    # print sample input
    input_batch, target_batch = next(iter(ds))

    print('Sample input :\n',mapper['id_to_str'](input_batch).numpy().decode('utf-8'))




