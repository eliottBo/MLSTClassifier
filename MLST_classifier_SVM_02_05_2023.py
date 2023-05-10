import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Import data:
train_cryptic = pd.read_csv('/Users/eliott/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Thesis_CR/data/mlst/MLST_dataset_known_cryptic.csv')
train_cryptic = train_cryptic.iloc[:,1:9]# get rid of index column

# Create training sets:
X = train_cryptic.iloc[:,0:7]
y = train_cryptic.iloc[:,7]

# split the training (2/3) and test(1/3) set and respect the proportion of the different outcomes in the training set (stratify):
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.33, stratify = y)

# Scaling the data
# X_train_scaled = scale(X_train)
# X_test_scaled = scale(X_test)

# CV:
param_grid = {'C': [0.5, 1, 10, 100], 'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf', 'sigmoid']}
grid_search = GridSearchCV(SVC(), param_grid, cv= 5, scoring = 'accuracy')
grid_results = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

svm = SVC()
clf_svm = svm.set_params(**grid_results.best_params_)
clf_svm.fit(X_train, y_train)

y_pred = clf_svm.predict(X_test)

# Set confusion matrix and output it:
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf_svm.classes_)
disp.plot()
plt.savefig('SVM_cm_knownvscryptic.svg', format= 'svg')
plt.show()

perf_report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(perf_report).transpose()
df.to_csv('perf_report_SVM_knownvscryptic.csv')
