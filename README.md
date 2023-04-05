# MLST-based clade classifiers

Here I implemented different ML classifiers (KNN and RF) that take a sequence type (ST) from a strain of *C.difficile* or *C-difficile*-like and output a clade prediction. This is to make it easier and faster to predict a clade instead of doing it with phylogenetic tree building using STs. These classifiers are far from perfect but with a bit more work they could be useful!
Inspired from StatQuest methodologiey (https://www.youtube.com/watch?v=q90UDEgYqeI&t=3327s)

## Steps:
- Loading dataset: training and testing sets are from public PubMLST database of C.difficile.
- Applying GridSearchCV to find the best parameters (for KNN)
- Training the classifier with best parameters
- Predicting clades on test set
- Printing report and confusion matrix

## What is next:
Add some more data about ST to refine the classifier and improve the Random Forest classifier to get better results.
