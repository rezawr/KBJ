import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_validate, train_test_split

from sklearn.svm import SVC

if __name__ == "__main__":
    result = []
    for x in ([1000, 10000, 20000]):
        df = pd.read_csv('datasets/movie.csv')
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=42, train_size=x)

        textclassifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('smote', SMOTE(random_state=12)),
            ('svc', SVC(kernel='linear', C=1, random_state=0, verbose=1))
        ])

        # Init First Train
        textclassifier.fit(X_train, y_train)

        # nb_pred = textclassifier.predict(X_test)
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
        scores = cross_validate(textclassifier, df['text'], df['label'], scoring=scoring)
        result.append({
            'dataset': x,
            'scores': scores
        })

    import pdb;pdb.set_trace()