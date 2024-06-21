import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle


datasets_dir = 'training_data/'
create_train_data = True


if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)


if create_train_data:
  #####################################################################
  #                       Creating the dataframe                      #
  #####################################################################

  download_folder = 'TON_IoT'
  csv_files = [f for f in os.listdir(download_folder) if f.endswith('.csv')]
  combined_df = pd.DataFrame()

  for csv_file in csv_files:
      print('concat', csv_file)
      file_path = os.path.join(download_folder, csv_file)
      df = pd.read_csv(file_path)
      combined_df = pd.concat([combined_df, df], ignore_index=True)
      del df

  #####################################################################
  #                       Feature Selection                           #
  #####################################################################

  # Choosing only 4 main classes to classify
  df = combined_df[combined_df['type'].isin(['normal', 'password', 'xss', 'scanning'])]
  print(df.keys())

  # Create a new DataFrame with only the columns that meet the criteria
  df = df.drop(['ts', 'src_ip', 'dst_ip', 'http_uri', 'http_referrer', 'http_version', 'uid'], axis=1)

  # Ensuring data type is uniform for src_bytes
  df = df.astype({'src_bytes': str})

  #####################################################################
  #                 Aggregating the data into groups                  #
  #####################################################################

  # Define the number of packets per group
  group_size = 10


  # Extract features and labels
  features = df.drop(['label', 'type'], axis=1)
  features = features.astype({'src_bytes': str})
  labels = df['type']

  # Ensure all features are numeric
  for column in features.columns:
      if features[column].dtype == 'object':
          features[column] = LabelEncoder().fit_transform(features[column])

  # Initialize lists for the aggregated features and labels
  aggregated_features = []
  aggregated_labels = []

  # Aggregate the data based on label groups
  unique_labels = labels.unique()
  for label in unique_labels:
      label_group = df[df['type'] == label]
      label_features = label_group.drop(['label', 'type'], axis=1)
      
      for i in range(0, len(label_group) - group_size + 1, group_size):
          group_features = label_features.iloc[i:i + group_size].values.flatten()  # Flatten the group features
          aggregated_features.append(group_features)
          aggregated_labels.append(label)


  # Create a DataFrame from the aggregated features and labels
  X_aggregated = pd.DataFrame(aggregated_features)
  print(X_aggregated)
  Y_aggregated = pd.Series(aggregated_labels)

  #####################################################################
  #                       Splitting the dataset                       #
  #####################################################################

  X_train, X_test, Y_train, Y_test = train_test_split(X_aggregated, Y_aggregated, test_size=0.4, random_state=42)

  #####################################################################
  #                       Saving to pickle files                      #
  #####################################################################

  # Save the datasets to pickle files
  with open(os.path.join(datasets_dir, 'X_train.pkl'), 'wb') as f:
      pickle.dump(X_train, f)
  with open(os.path.join(datasets_dir, 'X_test.pkl'), 'wb') as f:
      pickle.dump(X_test, f)
  with open(os.path.join(datasets_dir, 'Y_train.pkl'), 'wb') as f:
      pickle.dump(Y_train, f)
  with open(os.path.join(datasets_dir, 'Y_test.pkl'), 'wb') as f:
      pickle.dump(Y_test, f)

else:

  #####################################################################
  #                       Saving to pickle files                      #
  #####################################################################

  with open(os.path.join(datasets_dir, 'X_train.pkl'), 'rb') as f:
      X_train = pickle.load(f)
  with open(os.path.join(datasets_dir, 'X_test.pkl'), 'rb') as f:
      X_test = pickle.load(f)
  with open(os.path.join(datasets_dir, 'Y_train.pkl'), 'rb') as f:
      Y_train = pickle.load(f)
  with open(os.path.join(datasets_dir, 'Y_test.pkl'), 'rb') as f:
      Y_test = pickle.load(f)


#####################################################################
#                       Training the classifier                     #
#####################################################################

enc = OneHotEncoder(handle_unknown='ignore')
X_train = enc.fit_transform(X=X_train)
X_test = enc.transform(X=X_test)

clf = LinearSVC(random_state=42, C=1e-5)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print(f"Classifier Accuracy: {accuracy_score(Y_test, Y_pred)}\n")
print(f"Classification Report:\n {classification_report(Y_test, Y_pred)}\n")
