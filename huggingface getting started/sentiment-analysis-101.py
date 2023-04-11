from transformers import pipeline

classifier = pipeline('sentiment-analysis')

res = classifier("I have been waiting for a hugging face course my whole life.")

print(res)

# output : [{'label': 'POSITIVE', 'score': 0.9980935454368591}]

# Here we are using the default model : TFDistilBertForSequenceClassification
# but we can use any model we want like 
# -- saved locally 
# -- from hub 
# etc...

# Using a pipeline without specifying a model name and revision in production is not recommended.
# All model checkpoint layers were used when initializing TFDistilBertForSequenceClassification.
# All the layers of TFDistilBertForSequenceClassification were initialized from the model checkpoint at distilbert-base-uncased-finetuned-sst-2-english.
