
from transformers import pipeline

# The NER pipeline identifies entities such as persons, organizations or locations in a sentence

model = 'dbmdz/bert-large-cased-finetuned-conll03-english'
ner = pipeline(task='ner', aggregation_strategy="simple", model=model)


res = ner('Max Emilian Verstappen is a Dutch racing driver. He competes under the Dutch flag in Formula One with Red Bull Racing.')

print(res)