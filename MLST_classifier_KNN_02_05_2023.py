import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Import data:
train_cryptic = pd.read_csv('/Users/eliott/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Thesis_CR/data/mlst/MLST_dataset_known_cryptic.csv')
train_cryptic = train_cryptic.iloc[:,1:9]# get rid of index column
print(train_cryptic['mlst_clade'].value_counts())

# Create training sets:
X = train_cryptic.iloc[:,0:7]
y = train_cryptic.iloc[:,7]
# print(X.head(10))
# print(y.head(10))

# Split into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.33, stratify = y)

#KNN
# Create list of parameters for GridSearchCV and create the classifier object:
KNN_params = {'n_neighbors': range(1, 30, 2), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': range(1, 50, 5)}
knn = KNeighborsClassifier()

# Search for optimal parameters with GridSearch:
grid_search = GridSearchCV(estimator = knn, param_grid = KNN_params, n_jobs = 1, cv = 5, scoring = 'accuracy', error_score = 0)
grid_results = grid_search.fit(X_train, y_train)
print(grid_results.best_params_)

# Create classifier with best parameters from GridSearch:
final_model = knn.set_params(**grid_results.best_params_)
final_model.fit(X_train, y_train)

# Applying the final model on the test set:
y_pred = final_model.predict(X_test)

# Set the confusion matrix to assess the model:
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = final_model.classes_)
disp.plot()
#plt.savefig('KNN_cm_knownvscryptic.svg', format= 'svg')
#plt.show()

# Print a report with following metrics: precision, recall, F1-score
perf_report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(perf_report).transpose()
#print(df)
#df.to_csv('perf_report_KNN_knownvscryptic.csv')

# Save the model:
filename = 'finalized_KNN_model.sav'
joblib.dump(final_model, filename)
