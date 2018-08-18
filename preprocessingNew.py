import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
df = pd.read_csv("train.csv")
#df.info()
#print(df.describe())
df.plot.hist('Survived',by = 'Pclass', grid = 'False' , layout = [1,3], figsize = [10,3])