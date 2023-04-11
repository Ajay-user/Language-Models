from transformers import pipeline

sentiment = pipeline(task='sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

zeroshot = pipeline(task='zero-shot-classification', model='facebook/bart-large-mnli')

textgen = pipeline(task='text-generation', model='distilgpt2')



sentiment_results = sentiment(['I love HuggingFace', "I Love Tensorflow"])

zeroshot_results = zeroshot("We will learn NLP today", candidate_labels=['Education','Sports'])

textgen_results = textgen("Today we will learn", max_length=30, num_return_sequences=2)

print('SENTIMENT :',sentiment_results)
print('ZERO SHOT :',zeroshot_results)
print('TEXT GENERATION :',textgen_results)