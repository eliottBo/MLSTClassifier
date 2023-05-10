# Overview

## MLST-based clade classifiers
Here I implemented different ML classifiers (KNN, RF and SVM) that take a Multi-Locus Sequence Type (MLST) file from strains of *C.difficile* or *C-difficile*-like and output a clade prediction. This is to make it easier and faster to predict a clade instead of doing it with phylogenetic tree building using STs. These classifiers are not perfect (accuracies ~92% for RF and KNN) but with a bit more work they could be useful! I provided the pre-trained models of KNN and RF (.sav) for you to try them.
Inspired from StatQuest methodology (https://www.youtube.com/watch?v=q90UDEgYqeI&t=3327s).
I used Scikit-learn library

## Steps:
- Load dataset: training and testing sets are from public PubMLST database of *C.difficile*.
- Apply GridSearchCV to find the best parameters (for KNN) or do pruning for RF.
- Train the classifier with best parameters.
- Predict clades on test set
- Print report and confusion matrix

## Usage:
After downloading model.sav and MLST_classifier_EB.py and having your input file of MLST (see MLST_file_example.csv):
In the terminal write the following command:
```python3 MLST_classifier_EB.py path/to/input path/to/model.sav path/to/output```
