import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

sentences = dataset.iloc[:, 5].tolist()
labels = np.array(dataset.iloc[:, 0])
labels[labels == 4] = 1

training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(
    sentences,
    labels,
    test_size=0.2,
    random_state=31
)

training_labels_final = training_labels.astype('float32')
testing_labels_final = testing_labels.astype('float32')

vocab_size = 10000
embedding_dim = 100
max_length = 280
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

embeddings_index = {}

glove_path = "glove.6B.100d.txt"

with open(glove_path, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector

print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Assign the GloVe vector

model = tf.keras.models.Sequential()


embedding_layer = tf.keras.layers.Embedding(
    vocab_size,
    embedding_dim,
    weights=[embedding_matrix],
    input_length=max_length,
    trainable=False
)

model.add(embedding_layer)
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)))
model.add(tf.keras.layers.Dense(16, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

num_epochs = 10
history = model.fit(
    training_padded, training_labels_final,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels_final),
    callbacks=[callbacks]
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
