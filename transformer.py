import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

max_features = 10000
max_len = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_all = np.concatenate([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

X_train = X_all[:40000]
y_train = y_all[:40000]

X_test = X_all[40000:]
y_test = y_all[40000:]

# X_train = pad_sequences(X_train, maxlen=max_len)
# X_test = pad_sequences(X_test, maxlen=max_len)

word_index = imdb.get_word_index()
index_to_word = {index + 3 : word for word, index in word_index.items()}
index_to_word[0], index_to_word[1], index_to_word[2] = "<PAD>", "<START>", "<UNK>"

def decode(encoded_review):
	return " ".join([index_to_word.get(word, "<UNK>") for word in encoded_review])

X_train_text = [decode(review) for review in X_train]
X_test_text = [decode(review) for review in X_test]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

X_train_encodings = tokenizer(X_train_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="tf")
X_test_encodings = tokenizer(X_test_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="tf")

y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(
	{
		"input_ids": X_train_encodings["input_ids"],
		"attention_mask": X_train_encodings["attention_mask"]
	},
	y_train,
	validation_data=(
		{
			"input_ids": X_test_encodings["input_ids"],
			"attention_mask": X_test_encodings["attention_mask"]
		},
		y_test
	),
	epochs=5,
	batch_size=32
)

results = model.evaluate(
	{
		"input_ids": X_test_encodings["input_ids"],
		"attention_mask": X_test_encodings["attention_mask"]
	},
	y_test
)

print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")

model.save_pretrained("./models")
tokenizer.save_pretrained("./models")