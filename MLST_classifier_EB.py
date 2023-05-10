import pandas as pd
import sys
import joblib

# This script takes a MLST dataframe and predict the clade of 
# C.difficile including if it is cryptic based on the MLST labelled as cryptic

# First argument is the MLST df, one MLST per line, 8 columns (ST, 7 genes)
# Second argument is the path for the model file .sav
# Third argument is the output name
df= pd.read_csv(sys.argv[1])
model= joblib.load(sys.argv[2])# Import the pre-trained model
print(df.shape)

# Get rid of the ST column for prediction
X = df.iloc[:,1:8]
predicted_samples= pd.DataFrame(model.predict(X), columns= ['predicted_clade'])# Prediction
print(predicted_samples.value_counts())# Give an insight of the result

out_df= pd.concat([df,predicted_samples], axis= 'columns')# make a df with MLST and predictions
out_df['ST'] = df.iloc[:,0]# add back the ST column

out_df.to_csv(sys.argv[3])# Save the df into a csv file


