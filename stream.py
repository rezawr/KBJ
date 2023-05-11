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

import joblib
import os
import psutil
import time

def save_model(model, filename):
    joblib.dump(model, filename)

def delete_model(filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("The file does not exist")

def read_data(filename):
    # Read data from a file
    df = pd.read_csv(r'F:\Koding_Kopmputasi_berbasis_jaringan\sentiment amalysis dataset\movie.csv')
    return df['text'].tolist(), df['label'].tolist()

def record_performance():
    cpu_percent = psutil.cpu_percent()
    mem_info = psutil.virtual_memory()
    disk_io_counters = psutil.disk_io_counters()
    timestamp = time.time()
    # Write performance data to log file or database

# Example usage: Train an SVM model on your dataset and simulate a data stream of sentiment analysis results
data, labels = read_data(r'F:\Koding_Kopmputasi_berbasis_jaringan\sentiment amalysis dataset\movie.csv')
#df = pd.read_csv(r'F:\Koding_Kopmputasi_berbasis_jaringan\sentiment amalysis dataset\movie.csv')
# 20000  20000
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0, train_size=0.5)


# 2000 data split into 4 part
new_X_test = np.array_split(X_test, 10) #[10] = [2000. 2000 ...]
new_y_test = np.array_split(y_test, 10)
result = {
    'accuracy': [],
    'f1': [],
    'precision': [],
    'recall': []
}

for x in range(0, 10):
    textclassifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE(random_state=12)),
        ('svc', SVC(kernel='linear', C=1, random_state=0, verbose=1))
    ])
    # Start recording performance
    record_performance()

    textclassifier.fit(X_train, y_train)
    # After training the model in each iteration, you can save it like this:
    save_model(textclassifier, 'model_iteration_{}.joblib'.format(x))
    for i, data in enumerate(new_X_test[x]):
        svc_pred = textclassifier.predict(new_X_test[x])
        print('Data:', data)
        true_label = new_y_test[x][i]
        print('True label:', true_label)
        print('Predicted label:', svc_pred[0])
        print()
        #svc_pred = textclassifier.predict(new_X_test[x])
        # Record performance after each prediction
        record_performance()
        
    print('SVM classifier:', x)
    accuracy = accuracy_score(new_y_test[x], svc_pred)
    f1 = f1_score(new_y_test[x], svc_pred)
    precision = precision_score(new_y_test[x], svc_pred)
    recall = recall_score(new_y_test[x], svc_pred)

    result['accuracy'].append(accuracy)
    result['f1'].append(f1)
    result['precision'].append(precision)
    result['recall'].append(recall)
    # print('Accuracy:', accuracy_score(new_y_test, svc_pred))
    # print('F1 score:', f1_score(new_y_test, svc_pred, average='weighted'))
    # print('Precision:', precision_score(new_y_test, svc_pred, average='weighted'))
    # print('Recall:', recall_score(new_y_test, svc_pred, average='weighted'))
    # svc_cm = confusion_matrix(new_y_test, svc_pred)
    # sns.heatmap(svc_cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('SVM Classifier Confusion Matrix')
    # plt.show()

    srt = pd.Series(new_X_test[x])
    srr = pd.Series(svc_pred)

    X_train = X_train._append(srt)
    y_train = y_train._append(srr)

import pdb;pdb.set_trace()
# print(X_test)