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
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier

df = pd.read_csv('datasets/movie.csv')

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=0, train_size=0.5)


# 2000 data split into 4 part
new_X_test = np.array_split(X_test, 10)
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
        ('mnb', MultinomialNB(alpha =0.1))
    ])

    textclassifier.fit(X_train, y_train)
    nb_pred = textclassifier.predict(new_X_test[x])

    print('Naive Bayes classifier:', x)
    accuracy = accuracy_score(new_y_test[x], nb_pred)
    f1 = f1_score(new_y_test[x], nb_pred, average='weighted')
    precision = precision_score(new_y_test[x], nb_pred, average='weighted')
    recall = recall_score(new_y_test[x], nb_pred, average='weighted')

    result['accuracy'].append(accuracy)
    result['f1'].append(f1)
    result['precision'].append(precision)
    result['recall'].append(recall)
    # print('Accuracy:', accuracy_score(new_y_test, nb_pred))
    # print('F1 score:', f1_score(new_y_test, nb_pred, average='weighted'))
    # print('Precision:', precision_score(new_y_test, nb_pred, average='weighted'))
    # print('Recall:', recall_score(new_y_test, nb_pred, average='weighted'))
    # nb_cm = confusion_matrix(new_y_test, nb_pred)
    # sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('Naive Bayes Classifier Confusion Matrix')
    # plt.show()

    srt = pd.Series(new_X_test[x])
    srr = pd.Series(nb_pred)

    X_train = X_train._append(srt)
    y_train = y_train._append(srr)


import pdb;pdb.set_trace()
# print(X_test)