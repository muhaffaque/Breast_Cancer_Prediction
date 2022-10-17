#importing dataset
import pandas as pd
breast_cancer_data = pd.read_csv("breast_cancer_dataset.csv")
#print(breast_cancer_data.head())
#print(breast_cancer_data.shape)

#changing diagnosis column to boolean by labelencoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
breast_cancer_data.diagnosis = le.fit_transform(breast_cancer_data.diagnosis)
#print(breast_cancer_data.head())

X = breast_cancer_data.drop(labels = ['diagnosis'], axis=1)
y = breast_cancer_data['diagnosis']
#print(X.shape, y.shape)

#train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state=0)
#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#removing columns 'id' ,'Unnamed: 32' from dataset
X_train.drop(['id','Unnamed: 32'], axis = 1, inplace = True)
X_test.drop(['id','Unnamed: 32'], axis = 1, inplace = True)
#print(X_train.shape, X_test.shape)

import matplotlib.pyplot as plt
import seaborn as sns

#PEARSON CORRELATION
corr = X_train.corr()
#print(corr)
#highly correlated features
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix= dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>=threshold:
                col_name=corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr

corr_features = correlation(corr,0.80)
#print("The number of correlated features are: {} ".format(len(corr_features)))
#print("correlated-features are: {}".format(corr_features))

#dropping highly correlated features from X_train and X_test
X_train_non_corr = X_train.drop(corr_features,axis=1)
X_test_non_corr = X_test.drop(corr_features, axis=1)
#print(X_train_non_corr.shape,X_test_non_corr.shape,X_train.shape,X_test.shape)

#KNN model on default parameters
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier()  
classifier.fit(X_train_non_corr,y_train )


import pickle
filename='model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
