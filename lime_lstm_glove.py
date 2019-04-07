import numpy as np
import pandas as pd
import sklearn
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Embedding, Dropout, CuDNNLSTM
from keras import optimizers
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from lime.lime_text import LimeTextExplainer

# read dataset as pandas dataframe
df = pd.read_csv("tweet-dataset.csv", sep=",", encoding="utf-8")

texts_train, texts_test, y_train, y_test = train_test_split(df["SentimentText"].values.astype(str), df["Sentiment"].values, test_size=0.2, random_state=42)

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    """Computes the F1 Score
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())

vocab_size = 100000
maxlen = 25

# class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def fit(self, texts, texts_test, y=None):
#         self.fit_on_texts(texts)
#         return self
#
#     def transform(self, texts, y=None):
#         return np.array(self.texts_to_sequences(texts))
#
#
# sequencer = TextsToSequences(num_words=vocab_size)
#
# class Padder(BaseEstimator, TransformerMixin):
#
#     def __init__(self, maxlen=30):
#         self.maxlen = maxlen
#         self.max_index = None
#
#     def fit(self, X, y=None):
#         self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
#         return self
#
#     def transform(self, X, y=None):
#         X = pad_sequences(X, maxlen=self.maxlen)
#         X[X > self.max_index] = 0
#         return X
#
#
# padder = Padder(maxlen)

batch_size = 128
max_features = vocab_size

t = Tokenizer(num_words=max_features)
t.fit_on_texts(texts_train)
t.fit_on_texts(texts_test)
encoded_docs = t.texts_to_sequences(texts_train)
encoded_docs_test = t.texts_to_sequences(texts_test)

word_index = t.word_index

padded_docs = pad_sequences(encoded_docs, maxlen=maxlen)
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=maxlen)

embeddings_index = {}
with open("glove.twitter.27B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((max_features, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_features:
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

def create_model(max_features):
    model = Sequential()
    #model.add(Embedding(max_features, 128))
    e = Embedding(max_features, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(e)
    model.add(CuDNNLSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Dense(1, activation="sigmoid"))
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy", f1])
    return model

sklearn_cnn = KerasClassifier(build_fn=create_model, epochs=1, batch_size=batch_size,
                              max_features=max_features, verbose=1)

pipeline = make_pipeline(sklearn_cnn) #squencer, padder,

pipeline.fit(padded_docs, y_train) #texts_train, y_train

y_preds = pipeline.predict(padded_docs_test) #texts_test

print("y_test: ", y_test)
print("y_preds: ", y_preds)

# print("f1 score: ", sklearn.metrics.f1_score(y_test, y_preds, average="binary"))
# print("acc: ", sklearn.metrics.accuracy_score(y_test, y_preds, normalize=False))
print("classification_report: ", classification_report(y_test, y_preds))

explainer = LimeTextExplainer()

idx = 456

exp = explainer.explain_instance(texts_test[idx], pipeline.predict_proba, num_features=15, top_labels=2) # texts_test

exp.as_list()

out_as_html = exp.save_to_file("lstm_idx456.html")

text_sample = texts_test[idx] # texts_test

print("idx: ", idx)
print("true class: ", y_test[idx])
print("pred class: ", y_preds[idx])

