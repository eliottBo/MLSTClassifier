#Script to classify C.diff like species according to their ST. KNN is used as classifier
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


mlst_known = pd.read_csv('/Users/eliott/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Thesis_CR/scripts/ML/mlst_cryptic.csv')

mlst_known.drop(mlst_known.index[478], inplace= True, axis= 0)# Delete row 425 containing an odd mlst_clade number (103) and to be able to stratify later
mlst_known.reset_index(inplace = True, drop = True)# reset the indexes after deleting the column

# Separating features and labels:
X = mlst_known.iloc[:,1:8]# features column
y = mlst_known.iloc[:,8]# label column

# Print a summary of the different classes:
print(y.value_counts())

# split the training (2/3) and test(1/3) set and respect the proportion of the different outcomes in the training set (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.33, stratify = y)

# Create list of parameters for GridSearchCV and create the classifier object:
KNN_params = {'n_neighbors': range(1, 30, 2), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': range(1, 50, 5)}
knn = KNeighborsClassifier()

# Search for optimal parameters with GridSearch:
grid_search = GridSearchCV(estimator = knn, param_grid = KNN_params, n_jobs = 1, cv = 5, scoring = 'accuracy', error_score = 0)
grid_results = grid_search.fit(X_train, y_train)
#print(grid_results.best_params_)

# Create classifier with best parameters from GridSearch:
final_model = knn.set_params(**grid_results.best_params_)
final_model.fit(X_train, y_train)

# Applying the final model on the test set:
y_pred = final_model.predict(X_test)

# Set the confusion matrix to assess the model
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = final_model.classes_)
disp.plot()
# plt.savefig('KNN_cm_cv_no_scaling.svg', format= 'svg')
plt.show()

# Print a report with following metrics: precision, recall, F1-score
perf_report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(perf_report).transpose()
print(df)
# df.to_csv('perf_report_KNN_cv_noScaling.csv')
