from os import system
system('cls')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression


import pandas as pd

def clean(url):
    df=pd.read_csv(url)
    df['gender']=3
    df['gender']=df['Sex'].map({'female':0 , 'male':1}).astype(int)
    df['EmbarkCode'] = 3
    df['EmbarkCode'] = df['Embarked'].map({'S':0,'C':1,'Q':2})
    df['TotalParty'] = df['SibSp'] + df['Parch']
    df['FareAdjusted'] = df['Fare'] / 10.0
    df = df.drop(['Name','Ticket','Cabin','Sex','Embarked','SibSp','Parch','Fare'],axis=1)
    df.loc[(df.Age.isnull()),'Age']=0
    df.loc[(df.EmbarkCode.isnull()),'EmbarkCode'] = 3
    df.loc[(df.FareAdjusted.isnull()),'FareAdjusted'] = 1
    return df

df1=clean('train.csv')
print(df1.head())

df2=clean('test.csv')
print(df2.head())

X=df1.iloc[0::,2::].values
y=df1.iloc[0::,1].values

X_train ,X_test ,y_train ,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=LinearRegression()
clf.fit(X_train ,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)

y_pred =clf.predict(X_test)
plt.scatter(X_test, y_test, color = 'black')


plt.plot(X_test, y_pred, color = 'blue', linewidth=0.1)

plt.xlabel('factors')
plt.ylabel('suvived')

plt.show()
