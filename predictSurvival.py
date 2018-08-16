import preprocessing as pre
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
#convert our dataframe from pandas to numpy  and specify input and output
df = pre.createDataFrame("train.csv")
testdf = pre.createDataFrame("test.csv")
#testdf.info()
#print(df.head)
X = df.values
y = df['Survived'].values
# remove output field from input data
X = np.delete(X,1,axis = 1)
X_results = testdf.values
#splitting dataset into training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.3,random_state=0)

#Model 1 : using decision tree
decisionTree = tree.DecisionTreeClassifier(max_depth=5)
decisionTree.fit(X_train,y_train)
dtscore = decisionTree.score(X_test,y_test)
print("Decision Tree Score :" + str(dtscore))

#analyzing importance of features in decision tree from their entropy
#print(decisionTree.feature_importances_)

#Model 2 : using Random forest
randomForest = ensemble.RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train,y_train)
rfscore = randomForest.score(X_test,y_test)
print("Random Forest Score :" + str(rfscore))

#Model 3 : Gradient Boosting
gradientBoosting = ensemble.GradientBoostingClassifier(n_estimators=50)
gradientBoosting.fit(X_train,y_train)
gbscore = gradientBoosting.score(X_test,y_test)
print("Gradient boosting score: " + str(gbscore))

#predict on test data
gradientBoosting.fit(X,y)
y_results = gradientBoosting.predict(X_results)
output = np.column_stack((X_results[:,0],y_results))
df_results = pd.DataFrame(output.astype('int'),columns = ['PassengerId','Survived'])
df_results.to_csv('titanic_results.csv',index = False)

