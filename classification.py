import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

votes = pd.read_csv('votes.csv')
X = votes.drop(['party'], axis = 1)
y = votes['party']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

# classification non-linéaire K voisins les plus proches
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
y_pred_test_knn = knn.predict(X_test)
y_pred_test_knn[0:10]

# classification linéaire : Probabilité pour appartenance à classe
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_test_logreg = logreg.predict(X_test)
y_pred_test_logreg[0:10]

# Evaluation d'une classification avec une matrice de confusion du sous-module sklearn.metrics
matrice_confusion = confusion_matrix(y_test, y_pred_test_knn)
​print(matrice_confusion)
print(matrice_confusion[0,1], 'républicains qui sont en fait démocrates')
​(VN, FP), (FN, VP) = confusion_matrix(y_test, y_pred_test_knn)
​n = len(y_test)
print(n)
​print("accuracy knn : ", VN + VP / n)
print("precision knn : ", VP / VP + FP)
print("rappel knn : ", VP / VP + FN)

# Evaluation d'une classification avec une matrice de confusion avec la fonction pd.crosstab
​matrice_confusion_logreg = pd.crosstab(y_test, y_pred_test_logreg)
print(matrice_confusion_logreg)
​print(accuracy_score(y_test, y_pred_test_logreg))
print(precision_score(y_test, y_pred_test_logreg, pos_label = 'republican'))
print(recall_score(y_test, y_pred_test_logreg, pos_label = 'republican'))
​
# Evaluation d'une classification avec une matrice de confusion avec la fonction classification_report du sous-module sklearn.metrics
print(classification_report(y_test, y_pred_test_logreg))
​print(classification_report(y_test, y_pred_test_knn))

​# Evaluation d'une classification avec F1_score du sous-module sklearn.metrics
print(f1_score(y_test, y_pred_test_knn, pos_label = 'republican'))
print(f1_score(y_test, y_pred_test_logreg, pos_label = 'republican'))