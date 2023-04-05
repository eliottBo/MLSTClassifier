# Master's thesis MLST-based clade classifiers

Here I implemented different ML classifiers that take a sequence type (ST) from a strain of *C.difficile* or *C-difficile*-like and output a clade prediction. This is to make it easier and faster to predict a clade instead of doing it with phylogenetic tree building using STs.

## Steps:
- Loading dataset: training and testing sets are from public PubMLST database of C.difficile.
- Applying GridSearchCV to find the best parameters (for KNN)
- Training the classifier with best parameters
- Predicting clades on test set
- Printing report and confusion matrix

## What is next:
Add some more data about ST to refine the classifier and improve the Random Forest classifier to get better results.
