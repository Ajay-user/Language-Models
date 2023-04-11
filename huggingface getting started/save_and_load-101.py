
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# saving 
dir_name = './huggingface getting started/saved_models'
tokenizer.save_pretrained(save_directory=dir_name)
model.save_pretrained(dir_name)




