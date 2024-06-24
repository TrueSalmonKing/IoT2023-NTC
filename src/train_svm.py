import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


datasets_dir = 'training_data/'
create_train_data = False
load_classifier = True

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
  df = df[['proto', 'src_bytes', 'dst_bytes', 'conn_state', 
            'missed_bytes', 'src_ip_bytes', 
            'dst_ip_bytes', 'label', 'type']]

  # Ensuring data type is uniform for src_bytes
  df = df.astype({'src_bytes': str})

  #####################################################################
  #                       Splitting the dataset                       #
  #####################################################################

  X = df.drop(['label', 'type'], axis=1)
  Y = df['type']
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

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

  print('loading dataset files...')

  with open(os.path.join(datasets_dir, 'X_train.pkl'), 'rb') as f:
      X_train = pickle.load(f)
  with open(os.path.join(datasets_dir, 'X_test.pkl'), 'rb') as f:
      X_test = pickle.load(f)
  with open(os.path.join(datasets_dir, 'Y_train.pkl'), 'rb') as f:
      Y_train = pickle.load(f)
  with open(os.path.join(datasets_dir, 'Y_test.pkl'), 'rb') as f:
      Y_test = pickle.load(f)

  print('dataset files loaded')


#####################################################################
#                       Training the classifier                     #
#####################################################################

if load_classifier:
    print('transforming dataset files...')
    with open(os.path.join(datasets_dir, 'encoder.pkl'), 'rb') as f:
        enc = pickle.load(f)

    X_train = enc.transform(X_train)
    X_test = enc.transform(X=X_test)

    print('dataset files transformed')

    print('loading classifier...')

    with open(os.path.join(datasets_dir, 'clf.pkl'), 'rb') as f:
        clf = pickle.load(f)

    print('classifier loaded')

else:
    print('transforming dataset files...')

    enc = OneHotEncoder(handle_unknown='ignore')

    X_train = enc.fit_transform(X=X_train)
    with open(os.path.join(datasets_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(enc, f)

    X_test = enc.transform(X=X_test)

    print('dataset files transformed')

    print('training SVM...')

    clf = LinearSVC(class_weight='balanced', random_state=42, C=1e-2)
    clf.fit(X_train, Y_train)

    print('SVM trained')

    print('saving classifier...')

    with open(os.path.join(datasets_dir, 'clf.pkl'), 'wb') as f:
        pickle.dump(clf, f)

    print('classifier saved')

Y_pred = clf.predict(X_test)

print(f"Classifier Accuracy: {accuracy_score(Y_test, Y_pred)}\n")
#print(f"Classification Report:\n {classification_report(Y_test, Y_pred)}\n")


#####################################################################
#                   Plotting the confusion matrix                   #
#####################################################################

cm = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=clf.classes_, yticklabels=clf.classes_, cbar=False)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('plot.pdf')
