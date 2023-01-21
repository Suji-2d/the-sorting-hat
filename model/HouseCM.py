import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# creat data frame
griffindor_df = pd.DataFrame(data={
    'House':['Griffindor']*12500,
    'Extroversion': np.random.normal(35.02, 12.20, 12500), 
    'Agreeableness':np.random.normal(40.51, 7.35, 12500),
    'Conscientiousness':np.random.normal(35.88, 7.13, 12500),
    'Neuroticism':np.random.normal(24.66, 11.17, 12500),
    'Openness':np.random.normal(42.03, 6.32, 12500)}
)
hufflepuff_df = pd.DataFrame(data={
    'House':['Hufflepuff']*12600,
    'Extroversion': np.random.normal(30.79, 13.16, 12600), 
    'Agreeableness':np.random.normal(42.81, 7.27, 12600),
    'Conscientiousness':np.random.normal(38.76, 6.17, 12600),
    'Neuroticism':np.random.normal(21.00, 12.13, 12600),
    'Openness':np.random.normal(39.41, 9.33, 12600)}
)
ravenclow_df = pd.DataFrame(data={
    'House':['Ravenclow']*21400,
    'Extroversion': np.random.normal(29.52, 11.17, 21400), 
    'Agreeableness':np.random.normal(39.41, 7.35, 21400),
    'Conscientiousness':np.random.normal(36.76, 6.17, 21400),
    'Neuroticism':np.random.normal(25.75, 11.17, 21400),
    'Openness':np.random.normal(44.98, 8.23, 21400)}
)
slytherin_df = pd.DataFrame(data={
    'House':['Slytherin']*8700,
    'Extroversion': np.random.normal(27.42, 11.17, 8700), 
    'Agreeableness':np.random.normal(36.17, 10.22, 8700),
    'Conscientiousness':np.random.normal(33.70, 6.10, 8700),
    'Neuroticism':np.random.normal(28.97, 12.20, 8700),
    'Openness':np.random.normal(41.51, 7.35, 8700)}
)

house_df = pd.concat([griffindor_df,hufflepuff_df,ravenclow_df,slytherin_df])

# split X and y into training and testing sets
X = house_df.drop(['House'], axis=1)

y = house_df['House']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test
#print(X_train.shape, X_test.shape)

# the GaussianNB model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
#y_pred = gnb.predict(X_test)
#print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

pickle.dump(gnb, open('../houseModel.pkl','wb'))



#Decision tree model
# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# y_pred = dtc.predict(X_test)

# Random forest model
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)

# KNN model
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

