import pandas as pd
df = pd.read_csv("train.csv",header = 0)
# dropping columns that might not contribute to prediction like name,ticket no ,cabin etc
cols = ['Name','Ticket','Cabin']
df = df.drop(cols,axis = 1) #axis = 1 means column, axis = 0 means row
#drop all rows that has missing  NaN
#df = df.dropna()
# quantify all categorical variables or variables with more than 2 options
dummy = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
    dummy.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummy,axis=1)
#titanic_dummies.info()
df = pd.concat((df,titanic_dummies),axis=1)
# remove duplicate columns ie Pclass,Sex,Embarked
df = df.drop(cols,axis = 1)
# handling missing age values using interpolate
df['Age'] = df['Age'].interpolate()


