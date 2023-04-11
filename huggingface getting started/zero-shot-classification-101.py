
from transformers import pipeline

classifier = pipeline(task='zero-shot-classification', model="facebook/bart-large-mnli")


# We can pass this model a text without knowing the corresponding label
# then we put different candidate labels


res = classifier("This is a course about LLMs", candidate_labels=['education', 'politics', 'business'])

print(res)

# Zero shot classification pipeline is a more general text classification pipeline
# it allows you to provide the labels you want