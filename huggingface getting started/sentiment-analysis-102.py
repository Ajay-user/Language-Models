from transformers import pipeline

# distilbert-base-uncased-finetuned-sst-2-english
classifier = pipeline(task='sentiment-analysis',model='distilbert-base-uncased-finetuned-sst-2-english')


#  we can pass multiple text to the object returned by a pipeline 
res = classifier(["I have been waiting for a hugging face course my whole life.",'I hate dancing'])

print(res)

# output : [{'label': 'POSITIVE', 'score': 0.9980935454368591}]

# Here we are using the default model : TFDistilBertForSequenceClassification
# but we can use any model we want like 
# -- saved locally 
# -- from hub 
# etc...
