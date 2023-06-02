import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from imblearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_validate, train_test_split

from sklearn.svm import SVC


if __name__ == "__main__":
    result = []

    # Looping to find how number of datasets impacted on accuracy
    # Load dataset
    df = pd.read_csv('datasets/movie.csv')

    # Splitting Dataset
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=42, train_size=1000)

    # Build the model
    textclassifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svc', SVC(kernel='linear', C=1, random_state=0, verbose=1))
    ])

    # Train the data
    textclassifier.fit(X_train, y_train)

    # Scoring parameter that we want to analysis
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    # 10 fold cross validation
    scores = cross_validate(textclassifier, df['text'], df['label'], scoring=scoring)

    # save the result of prediction into variabel result
    result.append({
        'dataset': x,
        'scores': scores
    })

    # Just tracing
    import pdb;pdb.set_trace()