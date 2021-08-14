import numpy as np
import pandas as pd
import nltk
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

path = "data\\ai.txt"

list_sentence=[]
with open(path, encoding="utf8") as f:
    train_lines = f.readlines()
    for line in train_lines:
        line = line.split('__eou__')
        for i in range(len(line)):
            line[i] = line[i].strip()
        list_sentence.append(line)

get_Question = []
for i in list_sentence: get_Question.append(i[0])

get_Answer = []
for i in list_sentence: get_Answer.append(i[1])
    
labels = []
for i in range(len(get_Answer)): labels.append(i)

df = pd.DataFrame({"questions": get_Question, "answers": get_Answer, "label": labels})

num_classes = len(get_Question)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(labels)
labels = lbl_encoder.transform(labels)

vocab_size = 1600
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"
token = Tokenizer(num_words=vocab_size, oov_token=oov_token)
token.fit_on_texts(get_Question)
word_index = token.word_index
sequences = token.texts_to_sequences(get_Question)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

with open('token.pickle', 'wb') as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(LSTM(200))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()

epochs = 500
history = model.fit(padded_sequences, np.array(labels), epochs=epochs)

model.save("model.h5")