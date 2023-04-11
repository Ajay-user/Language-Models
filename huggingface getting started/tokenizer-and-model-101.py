from transformers import pipeline

# lets import generic tokenizer and model
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

pipe = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer)

res = pipe('This is awesome.')


print(res)