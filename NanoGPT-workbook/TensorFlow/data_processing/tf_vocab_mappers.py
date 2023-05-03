import tensorflow as tf
import time
import logging


def get_character_maps(vocabulary:str)->dict[str, tf.keras.layers.Layer]:
    '''
    Creates mappers that maps input characters to integer-ids and vice-versa 

    Outputs a mapper object\n
    {\n
    `'char_to_id'`: ## maps characters to interger ids ## ,\n
    `'id_to_char'`:  ## maps interger-ids to characters ## , \n
    `'id_to_str'`:   ## maps interger-ids to string ## \n
    }

    '''
    logging.info('Creating-vocab-maps...')
    char_vocab = sorted(set(vocabulary))
    char_to_id = tf.keras.layers.StringLookup(mask_token=None, vocabulary=char_vocab)
    id_to_char = tf.keras.layers.StringLookup(mask_token=None, vocabulary=char_to_id.get_vocabulary(), invert=True)
    id__to_str = lambda id: tf.strings.reduce_join(inputs=id_to_char(id), separator='', axis=-1)
 
    return {'char_to_id':char_to_id ,'id_to_char':id_to_char, 'id_to_str':id__to_str}




if __name__ == "__main__":


    from tf_data import get_data

    logging.basicConfig(level=logging.INFO)

    start = time.time()

    data = get_data("./NanoGPT-workbook/data/tinyshakespeare.txt")
    character_maps = get_character_maps(data)
    
    end = time.time()

    print(f'TIME COST : {end-start}')
    print('Vocab size :',character_maps['char_to_id'].vocabulary_size())