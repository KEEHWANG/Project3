#diabetes 예측 모델
import pandas as pd

filename = 'diabetes.csv'
df = pd.read_csv(filename)

print(df.shape)
df.head()

#결측치,데이터타입, 데이터 분포 확인

print(df.dtypes)
print(df.isnull().sum())

baseline = df['Outcome'].value_counts(normalize=True)
baseline

#전처리 (비정상적인 0값을 중앙값으로 대체)

print(df['Insulin'].median()) 
print(df['SkinThickness'].median())

df['Insulin'] = df['Insulin'].replace(0,30.5)
df['SkinThickness'] = df['SkinThickness'].replace(0,23)

df.head()

#X,y 데이터 나누기

X = df.drop(columns = ['Outcome']) 
y = df['Outcome']

X.shape

#Train/Validation/Test셋 나누어주기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, test_size=0.2)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

X_train.head()

#모델링

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from category_encoders import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn. preprocessing import OneHotEncoder
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


pipeline_rf = make_pipeline(
    OrdinalEncoder(), 
    SimpleImputer(strategy='most_frequent'), 
    RandomForestClassifier(n_estimators = 200, random_state=100, n_jobs=-1, oob_score=True)
)

pipeline_rf.fit(X_train, y_train)

#randomizedsearchCV

dists = {  
    'simpleimputer__strategy': ['most_frequent', 'median'], 
    'randomforestclassifier__n_estimators': randint(50, 500), 
    'randomforestclassifier__max_depth': [5, 10, 15, 20, None], 
    'randomforestclassifier__max_features': uniform(0, 1) 
}

clf = RandomizedSearchCV(
    pipeline_rf, 
    param_distributions=dists, 
    n_iter=10, 
    cv=3, 
    scoring='f1',  
    verbose=1,
    n_jobs=-1
)

clf.fit(X_train, y_train)
print("Optimal Hyperparameter:", clf.best_params_)
print("AUC:", clf.best_score_)


#검증 및 테스트 예측결과

print('검증 정확도', pipeline_rf.score(X_val, y_val))
print('테스트 예측 결과', pipeline_rf.score(X_test, y_test))

y_pred = pipeline_rf.predict(X_test)
print('테스트셋 f1_score', f1_score(y_test, y_pred))

#피클파일 생성

import pickle

with open('modeling.pkl', 'wb') as pickle_file:
    pickle.dump(pipeline_rf, pickle_file)
