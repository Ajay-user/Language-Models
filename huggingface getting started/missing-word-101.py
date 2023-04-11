from transformers import pipeline

# FILL MASK pipeline will predict missing words in a sentence

unmasker = pipeline(task='fill-mask')


res = unmasker('This course will teach you everything about <mask> models', top_k=2)


print(res)

# Sample output
# [{'score': 0.21518942713737488, 'token': 30412, 'token_str': ' mathematical', 'sequence': 'This course will teach you everything about mathematical models'},
#  {'score': 0.03962624445557594, 'token': 27930, 'token_str': ' predictive', 'sequence': 'This course will teach you everything about predictive models'}]