import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
df = pd.read_csv("train.csv")
#df.info()
#print(df.describe())
df.hist('Survived',by = 'Pclass', grid = 'False' , layout = [1,3], figsize = [10,3])
df.hist('Survived',by = 'Sex', figsize = [10,3])
df.hist('Survived',by = 'Embarked' , layout = [1,3], figsize = [10,3])
df.hist('Age', by = ['Survived','Sex'] , layout = [1,4], figsize = [20,3])
#plot.show()
#print(df.isnull().sum())
#Handing Missing Values
#Method 1 : replace missing age value with average
'''
avg = np.average(df['Age'].fillna(value = 0))
df['Age'].fillna(value = avg,inplace = True)
print(df['Age'].describe())
'''
df['Title'] = df['Name'].str.lower().str.extract('([a-z]*\.)',expand = True)

'''
print(set(df['Title']))
print(df['Age'].isnull().sum())
print(((df['Age'].isnull()) & (df['Title'] == 'mr.')).sum())
print(((df['Age'].isnull()) & (df['Title'] == 'mrs.')).sum())
print(((df['Age'].isnull()) & (df['Title'] == 'miss.')).sum())
print(((df['Age'].isnull()) & (df['Title'] == 'master.')).sum())
print(((df['Age'].isnull()) & (df['Title'] == 'dr.')).sum())
'''
avg_mr = df[((df['Title'] == 'mr.')&(df['Age'].isnull() == False))]['Age'].median()
avg_mrs = df[((df['Title'] == 'mrs.')&(df['Age'].isnull() == False))]['Age'].median()
avg_master = df[((df['Title'] == 'master.')&(df['Age'].isnull() == False))]['Age'].median()
avg_miss = df[((df['Title'] == 'miss.')&(df['Age'].isnull() == False))]['Age'].median()
avg_dr = df[((df['Title'] == 'dr.')&(df['Age'].isnull() == False))]['Age'].median()
df.loc[((df['Title'] == 'dr.') & (df['Age'].isnull() == True)),'Age'] = avg_dr
df.loc[((df['Title'] == 'mr.') & (df['Age'].isnull() == True)),'Age'] = avg_mr
df.loc[((df['Title'] == 'master.') & (df['Age'].isnull() == True)),'Age'] = avg_master
df.loc[((df['Title'] == 'mrs.') & (df['Age'].isnull() == True)),'Age'] = avg_mrs
df.loc[((df['Title'] == 'miss.') & (df['Age'].isnull() == True)),'Age'] = avg_miss
#print(df['Age'].describe())
age_mean = df['Age'].mean()
age_sd = df['Age'].std()
df['age_norm'] = ((df['Age'] - age_mean)/age_sd)
#print(df['age_norm'].describe())
# one hot encoding
df['is_male'] = pd.get_dummies(df['Sex'],drop_first=True)
bins = [0, 15, 25, 50, 100]
df['age_group'] = pd.cut(df['Age'],bins)
df[['age15','age25','age50','age100']] = pd.get_dummies(df['age_group'])
df.info()
df.hist('Survived',by = ['Sex','age_group'], layout = [2,4], figsize = [10,10])
plot.show()
