import tensorflow as tf
import requests
import pathlib
import logging



def get_data(url:str)->str:
    '''
    Download the text data or read the text file from a local path: 
    If the url starts with `http` this method will download the data from that url,
    else if the url is a local path, this will read the text file 
    '''
    if url.startswith('http'):
        logging.info("Downloading-Data...")
        data = requests.get(url=url).text
    else:
        logging.info("Reading-Data...")
        data = pathlib.Path(url).read_text()
    return data






if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    url = "./NanoGPT-workbook/data/tinyshakespeare.txt"
    data = get_data(url)
    print(data[:250])

    
