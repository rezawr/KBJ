import threading
import time
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC


# Duration in second
duration = 300
duration_training = 60

# Thread 
def validation(stop_event):
    global X_train, y_train, result
    while not stop_event.is_set():
        print("TEST")
        data = df.sample(random.randint(1, 100))

        nb_pred = textclassifier.predict(data['text'])

        X_train = X_train._append(data['text'])
        y_train = y_train._append(pd.Series(nb_pred))

        result['accuracy'].append(accuracy_score(data['label'], nb_pred))
        result['f1'].append(f1_score(data['label'], nb_pred, average='weighted'))
        result['precision'].append(precision_score(data['label'], nb_pred, average='weighted'))
        result['recall'].append(recall_score(data['label'], nb_pred, average='weighted'))
        time.sleep(1)

def train(stop_event):
    global textclassifier
    while not stop_event.is_set():
        print("TRAIN")
        textclassifier.fit(X_train, y_train)
        time.sleep(duration_training)


if __name__ == "__main__":
    df = pd.read_csv('datasets/movie.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=0, train_size=0.01)
    data = df.sample(random.randint(1, 100))
    result = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    textclassifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE(random_state=12)),
        ('svc', SVC(kernel='linear', C=1, random_state=0, verbose=1))
    ])

    # Init First Train
    textclassifier.fit(X_train, y_train)

    event_stop = threading.Event()
    t1 = threading.Thread(target=train, args=(event_stop,)).start()
    t2 = threading.Thread(target=validation, args=(event_stop,)).start()

    time.sleep(duration)
    event_stop.set()
    print("Done!")
    import pdb;pdb.set_trace()