from flask import Flask, Blueprint, render_template, request
from . import db
from flask_login import login_required, current_user

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

path = "project\\data\\ai.txt"
list_sentence=[]
with open(path, encoding="utf8") as f:
    train_lines = f.readlines()
    for line in train_lines:
        line = line.split('__eou__')
        for i in range(len(line)):
            line[i] = line[i].strip()
        list_sentence.append(line)

get_Question = []
for i in list_sentence:
    get_Question.append(i[0])

get_Answer = []
for i in list_sentence:
    get_Answer.append(i[1])
    
labels = []
for i in range(len(get_Answer)):
    labels.append(i)

df = pd.DataFrame({"questions": get_Question, "answers": get_Answer, "label": labels})

def predict_class(msg):
    model = keras.models.load_model('project/model.h5')

    with open('project/token.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    with open('project/label.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    max_len = 20

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([msg]),
                                         truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    new_tag = tag.astype(np.int)
    for i in range(len(df)):
        if i == new_tag:
            responses = df['answers'][i]
    return responses


def chatbot_response(msg):
    res = predict_class(msg)
    return res

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html', name=current_user.name)

@main.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)