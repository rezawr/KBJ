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
    ('mnb', GradientBoostingClassifier())
])

textclassifier.fit(X_train, y_train)

gb_pred = textclassifier.predict(X_test)

print('Gradient Boosting classifier:')
print('Accuracy:', accuracy_score(y_test, gb_pred))
print('F1 score:', f1_score(y_test, gb_pred, average='weighted'))
print('Precision:', precision_score(y_test, gb_pred, average='weighted'))
print('Recall:', recall_score(y_test, gb_pred, average='weighted'))
nb_cm = confusion_matrix(y_test, gb_pred)
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Gradient Boosting Classifier Confusion Matrix')
plt.show()


""""
Result:
Accuracy: 0.8066
F1 score: 0.8060178657646946
Precision: 0.8089426085600686
Recall: 0.8066
"""

