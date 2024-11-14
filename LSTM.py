import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np


# We only keep the most common 10,000 words
max_features = 10000
max_len = 500  # Maximum length of each comment

# Load the data and convert the reviews into integer-encoded sequences
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Combine all data
x_full_train = np.concatenate([x_train, x_test])
y_full_train = np.concatenate([y_train, y_test])

# The first 40,000 records are used as the new training set, and the remaining 10,000 records are used as the new test set.
x_train_new = x_full_train[:40000]
y_train_new = y_full_train[:40000]

x_test_new = x_full_train[40000:]
y_test_new = y_full_train[40000:]

# Pad or truncate the merged data
x_train_new = sequence.pad_sequences(x_train_new, maxlen=max_len)
x_test_new = sequence.pad_sequences(x_test_new, maxlen=max_len)

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
model.add(LSTM(units=128, return_sequences=True))  # LSTM layer
model.add(Dropout(0.4))
model.add(LSTM(units=128, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(1, activation='sigmoid'))  # binary classfication


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


batch_size = 64
epochs = 2

history = model.fit(x_train_new, y_train_new,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test_new, y_test_new))


score, acc = model.evaluate(x_test_new, y_test_new, batch_size=batch_size)
print(f'Test score: {score}')
print(f'Test accuracy: {acc}')

