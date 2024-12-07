import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Add
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# We only keep the most common 10,000 words
max_features = 10000
max_len = 500  # Maximum length of each comment
embedding_dim = 128  # Embedding Dimensions
num_heads = 4  # MultiHeadAttention number
ff_dim = 128  # Hidden layer dimensions of feedforward networks
num_transformer_blocks = 2  # Number of Transformer Blocks
dropout_rate = 0.4  # Dropout

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

# Define Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),  # Output back to embed_dim
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # First layer normalization
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # Second layer normalization
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        # Multihead attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection + normalization
        # Feedforward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Residual connection + normalization

# Define Embedding + Transformer model
def create_transformer_model():
    # Input layer
    inputs = Input(shape=(max_len,))

    # Embedding layer
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len)
    x = embedding_layer(inputs)

    # Several Transformer Blocks
    for _ in range(num_transformer_blocks):
        transformer_block = TransformerBlock(embed_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)
        x = transformer_block(x)

    # Global average pooling layer, dimensionality reduction
    x = GlobalAveragePooling1D()(x)

    # Dropout
    x = Dropout(dropout_rate)(x)

    # Fully connected layer for binary classification
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
model = create_transformer_model()

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


batch_size = 64
epochs = 3

# Train the model
history = model.fit(
    x_train_new, y_train_new,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test_new, y_test_new)
)
# Evaluate the model
score, acc = model.evaluate(x_test_new, y_test_new, batch_size=batch_size)
print(f'Test score: {score}')
print(f'Test accuracy: {acc}')

# Generate predictions on the test set
y_pred_prob = model.predict(x_test_new, batch_size=batch_size)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Confusion Matrix
mat = confusion_matrix(y_test_new, y_pred)

# Heatmap Visualization
sns.set(rc={'figure.figsize': (8, 8)})
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test_new, y_pred, target_names=['Negative', 'Positive']))
