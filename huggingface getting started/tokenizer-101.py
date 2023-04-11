from transformers import pipeline
from transformers import AutoTokenizer


model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)


input_text = 'Hello this is Transfomers 101 and its awesome ! '

res = tokenizer(input_text)
print('Tokenizer output :',res)


toks = tokenizer.tokenize(input_text)
print("Tokens :",toks)

ids = tokenizer.convert_tokens_to_ids(toks)
print('IDs :', ids)

ids_to_toks = tokenizer.decode(ids)
print('Decode :', ids_to_toks)

