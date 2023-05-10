import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Import data:
train_cryptic = pd.read_csv('/Users/eliott/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Thesis_CR/data/mlst/MLST_dataset_known_cryptic.csv')
train_cryptic = train_cryptic.iloc[:,1:9]# get rid of index column

# Split features from labels:
X = train_cryptic.iloc[:,0:7]
y = train_cryptic.iloc[:,7]
# print(X.head(10))
# print(y.head(10))

## RF
# Split into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.33, stratify = y)

# Pruning:
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)

# Extract alpha values:
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts = []

# Perform CV with 5 trees for each alpha value and plot the mean accuracy:
alpha_loop_values = []
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(ccp_alpha = ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv = 5)# compute the 5 trees with the same alpha value
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])# list of the alpha, mean accuracy of 5 trees and sd

alpha_results = pd.DataFrame(alpha_loop_values, columns= ['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x = 'alpha', y = 'mean_accuracy', yerr = 'std', marker = 'o', linestyle = '--')
plt.savefig('RF_cv_accuracies_knownvscryptic.svg', format = 'SVG')
plt.show()

# Extract the alpha value corresponding to the best accuracy:
#print(alpha_results[alpha_results['mean_accuracy'] > 0.95])
optimal_accuracy = alpha_results['mean_accuracy'].max()
optimal_alpha = alpha_results[alpha_results['mean_accuracy'] == optimal_accuracy]['alpha'].values[0]

# Initialize the classifier with the best found alpha and predict:
clf_dt_pruned = DecisionTreeClassifier(ccp_alpha= optimal_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

y_pred = clf_dt_pruned.predict(X_test)

# Build and display the confusion matrix:
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf_dt_pruned.classes_)
disp.plot()
plt.savefig('RF_cm_02_05_knownvscryptic.svg', format= 'svg')
plt.show()

# Display the performance report:
print(classification_report(y_test, y_pred))
perf_report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(perf_report).transpose()
df.to_csv('perf_report_RF_02_05_knownvscryptic.csv')

# Plot the decision tree:
plt.figure(figsize = (15, 7.5))
plot_tree(clf_dt_pruned, filled = True, rounded = True, class_names = ['1','2','3','4','5', 'cryptic'], feature_names = X.columns)
plt.savefig('tree_RF_pruned_knownvscryptic.svg', format = 'svg')
plt.show()

# Save the model:
filename = 'finalized_RF_model.sav'
joblib.dump(clf_dt_pruned, filename)