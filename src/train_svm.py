import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import random
import numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


download_dataset = False


#####################################################################
#                       Creating the dataframe                      #
#####################################################################

download_folder = 'TON_IoT'
csv_files = [f for f in os.listdir(download_folder) if f.endswith('.csv')]
print(csv_files)
combined_df = pd.DataFrame()

for csv_file in csv_files:
    print('concat', csv_file)
    file_path = os.path.join(download_folder, csv_file)
    df = pd.read_csv(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)
    del df

print(combined_df)
#combined_df.to_csv('combined_dataset.csv', index=False)
df = combined_df

# Choosing only 4 main classes to classify
df = df[df['type'].isin(['normal', 'password', 'xss', 'scanning'])]

print(df.keys())


#####################################################################
#                       Splitting the dataset                       #
#####################################################################


X = df[df.keys()].drop(['label', 'type'], axis=1)
X = X.astype({'src_bytes': str})
print(len(X))
le = LabelEncoder()

Y = df['type']
print(len(Y))

for column in X.columns:
  if X[column].dtype == 'object':
    X[column] = le.fit_transform(X[column])

print(X.nunique())

# Define the threshold for the maximum number of unique values
threshold = [1, 50]

# Compute the number of unique values for each column
unique_counts = X.nunique()

# Filter columns based on the threshold
columns_to_keep = unique_counts[(unique_counts > threshold[0]) & (unique_counts <= threshold[1])].index

# Create a new DataFrame with only the columns that meet the criteria
X_new = X[columns_to_keep]
print(X_new.columns)
print('lengths', len(X), len(Y))


X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2)
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

#####################################################################
#                       Training the classifier                     #
#####################################################################


clf = RandomForestClassifier(class_weight='balanced')

clf.fit(X_train, Y_train)


Y_pred = clf.predict(X_test)

print(f"Classifier Accuracy: {accuracy_score(Y_test, Y_pred)}\n")
print(f"Classification Report:\n {classification_report(Y_test, Y_pred)}\n")
