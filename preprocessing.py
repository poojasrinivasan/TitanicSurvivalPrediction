import numpy as np
import pandas as pd
def createDataFrame(input):
    df = pd.read_csv(input,header = 0)
    # dropping columns that might not contribute to prediction like name,ticket no ,cabin etc
    cols = ['Ticket', 'Cabin','Name']
    df = df.drop(cols, axis=1)  # axis = 1 means column, axis = 0 means row
    # drop all rows that has missing  NaN
    # df = df.dropna()
    # quantify all categorical variables or variables with more than 2 options
    dummy = []
    cols = ['Pclass', 'Sex', 'Embarked']
    for col in cols:
        dummy.append(pd.get_dummies(df[col]))
    titanic_dummies = pd.concat(dummy, axis=1)
    # titanic_dummies.info()
    df = pd.concat((df, titanic_dummies), axis=1)
    # remove duplicate columns ie Pclass,Sex,Embarked
    df = df.drop(cols, axis=1)
    # handling missing age values using interpolate
    df['Age'] = df['Age'].interpolate()
    df['level'] = np.where(df['Age']>=18 ,'adult','child')
    dummyLevelDf = pd.get_dummies(df['level'])
    df = pd.concat((df,dummyLevelDf),axis=1)
    df = df.drop(['level','Age','PassengerId'],axis = 1)
    df['Fare'] = df['Fare'].interpolate()
    return df
