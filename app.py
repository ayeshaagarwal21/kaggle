import numpy as np 
import pandas as pd
import os
print(os.listdir("/kaggle/input/insurance"))

df = pd.read_csv("/kaggle/input/insurance/insurance.csv")
df.head()

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['sex']=lb.fit_transform(df['sex'])
df['smoker']=lb.fit_transform(df['smoker'])
df['region']=lb.fit_transform(df['region'])
df.head(2)
x=df.drop(columns=['charges','smoker'])
y=df['smoker']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier 
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn.model_selection import GridSearchCV
param_grid ={
    'n_estimators': [100,200,300],
    'max_depth':[None,5,10,15],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['sqrt','log2']
}
grid_search=GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  #5-fold cross validation
    n_jobs=-1, #use all cpu core
    verbose=1,
    scoring='accuracy'
)
grid_search.fit(x_train,y_train)
#get the best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
