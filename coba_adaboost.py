import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier

df = pd.read_csv('datasets/movie.csv')
# print(df.head())

negative = df[df['label']==0]
positive = df[df['label']==1]
negative.shape, positive.shape


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=0)
# X_train.shape, X_test.shape

textclassifier = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('smote', SMOTE(random_state=12)),
    ('mnb', AdaBoostClassifier())
])

textclassifier.fit(X_train, y_train)

gb_pred = textclassifier.predict(X_test)

print('Adaboost classifier:')
print('Accuracy:', accuracy_score(y_test, gb_pred))
print('F1 score:', f1_score(y_test, gb_pred, average='weighted'))
print('Precision:', precision_score(y_test, gb_pred, average='weighted'))
print('Recall:', recall_score(y_test, gb_pred, average='weighted'))
nb_cm = confusion_matrix(y_test, gb_pred)
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Adaboost Classifier Confusion Matrix')
plt.show()


""""
Result:
Accuracy: 0.7995
F1 score: 0.7992620222239856
Precision: 0.800151194390893
Recall: 0.7995
"""

