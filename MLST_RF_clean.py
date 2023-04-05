import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Import data:
mlst_known = pd.read_csv('/Users/eliott/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Thesis_CR/scripts/ML/mlst_cryptic.csv')

mlst_known.drop(mlst_known.index[478], inplace= True, axis= 0)# Delete row 425 containing an odd mlst_clade number (103)
mlst_known.reset_index(inplace = True, drop = True)# reset the indexes

# Separate features from labels:
X = mlst_known.iloc[:,1:8]
y = mlst_known.iloc[:,8]

# Split the training (2/3) and test(1/3) set and respect the proportion of the different outcomes in the training set (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.33, stratify = y)

# Initialize classifier object and train it:
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)

# Generating decision tree:
plt.figure(figsize = (15, 7.5))
plot_tree(clf_dt, filled = True, rounded = True, class_names = ['1','2','3','4','5', 'unknown'], feature_names = X.columns)
#plt.savefig('tree_RF', format = 'svg')
plt.show()

# Test the classifier on the test set:
y_pred = clf_dt.predict(X_test)

# Print report on performance:
perf_report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(perf_report).transpose()
print(df)
#df.to_csv('perf_report_RF_noScaling.csv')

# Generate confusion martix:
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf_dt.classes_)
disp.plot()
#plt.savefig('RF_cm_noScaling.svg', format= 'svg')
plt.show()


