import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/wdbc.data', header=None)
                                 # Breast Cancer Wisconsin dataset

X, y = df.values[:, 2:], df.values[:, 1]

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(encoder.transform(['M', 'B']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

class MyStandard(BaseEstimator, TransformerMixin):
    "准备实现一下StandardScaler的基本功能，默认输入格式是numpy.array"
    def __init__(self):
        self.column = 0
        self.row = 0
        pass

    def fit(self, x, *_):
        self.mean = np.mean(x, axis=0)
        self.variance = np.var(x, axis=0)
        return self

    def transform(self, x, *_):
        return_x = (x - self.mean)/self.variance
        return return_x

pipe = Pipeline([('sc', MyStandard()),
                ('pca', PCA()),
                ('clf', RandomForestClassifier())
                ])
# 原始的版本
# pipe = Pipeline([('sc', StandardScaler()),
#                 ('pca', PCA()),
#                 ('clf', RandomForestClassifier())
#                 ])

param_grid = {
    'pca__n_components': [5, 10, 20],
    'clf__n_estimators': [5, 10, 20],
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

print('Test accuracy: %.3f' % search.score(X_test, y_test))

