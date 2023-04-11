
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification



model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)


inputs = ['This is awesome', 'I dont like romantic movies', 'I love MIT open-courses']

# batches
tok_batch = tokenizer(inputs, truncation=True, padding=True, return_tensors='tf')
# print(tok_batch)


outputs = model(tok_batch)
print('RAW OUTPUTS FROM MODEL :',outputs)

softmax_out = tf.nn.softmax(outputs.logits)
print('Softmax output :',softmax_out)

final_labels = tf.argmax(softmax_out, axis=1)
print('Model prediction :', final_labels)