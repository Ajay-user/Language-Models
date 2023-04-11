from transformers import pipeline

# The question answering pipeline extract answers to a question 
# from a given context


qa = pipeline(task='question-answering', model='distilbert-base-cased-distilled-squad')

res_1 = qa(question='who is max verstappen?', context='Max Emilian Verstappen is a Dutch racing driver and the 2021 and 2022 Formula One World Champion. He competes under the Dutch flag in Formula One with Red Bull Racing. Red Bull is now leading the Championship.')

# Providing the context and a question, the model will identify the span of text in the context
# containing the answer to the question

res_2 = qa(question='who won the Formula One World Championship in 2021?', context='Max Emilian Verstappen is a Dutch racing driver and the 2021 and 2022 Formula One World Champion. He competes under the Dutch flag in Formula One with Red Bull Racing. Red Bull is now leading the Championship.')

print(res_1)
print(res_2)