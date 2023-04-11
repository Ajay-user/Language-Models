from transformers import pipeline


gen = pipeline(task='text-generation', model='distilgpt2')


res = gen("In this course, we'll teach you how to", max_length=30, num_return_sequences=2)

print(res)

# Text generation pipeline use an input prompt to generate text
# The output is generated with a bit of randomness, so it changes each time you call the generator object
