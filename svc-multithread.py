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
duration = 301
duration_training = 60

# Thread
def validation(stop_event):
    global X_train, y_train, result, x
    while not stop_event.is_set():
        print("VALIDATION ", x)
        x += 1
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
    global y, textclassifier, run_classification
    while not stop_event.is_set():
        print("TRAINING ", y)
        y += 1
        # textclassifier.fit(X_train, y_train)
        tmp_classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('smote', SMOTE(random_state=12)),
            ('svc', SVC(kernel='linear', C=1, random_state=0, verbose=1))
        ])

        # Init First Train
        tmp_classifier.fit(X_train, y_train)
        textclassifier = tmp_classifier
        time.sleep(duration_training)


if __name__ == "__main__":
    x = 1
    y = 1
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
    run_classification = True
    t1 = threading.Thread(target=validation, args=(event_stop,)).start()

    time.sleep(duration_training)
    t2 = threading.Thread(target=train, args=(event_stop,)).start()
    # t2.join()

    time.sleep(duration)
    event_stop.set()
    print("Done!")

    x_axis = list(range(1, x))

    # accuracy
    plt.plot(x_axis, result['accuracy'], label="Accuracy")
    plt.xlabel('Time Series')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy')

    # F1 Score
    plt.plot(x_axis, result['f1'], label="F1 Score")
    plt.xlabel('Time Series')
    # plt.ylabel('F1 Score')
    # plt.title('F1 Score')

    # Precision
    plt.plot(x_axis, result['precision'], label="Precision")
    plt.xlabel('Time Series')
    # plt.ylabel('Precision')
    # plt.title('Precision')

    # Recall
    plt.plot(x_axis, result['recall'], label="Recall")
    plt.xlabel('Time Series')
    plt.ylabel('Percentage')
    plt.title('Time Series Graph')
    plt.legent()
    plt.show()
    import pdb;pdb.set_trace()