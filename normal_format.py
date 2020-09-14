import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/wdbc.data', header=None)
                                 # Breast Cancer Wisconsin dataset]


X, y = df.values[:, 2:], df.values[:, 1]
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(encoder.transform(['M', 'B']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
# X_train_trans = scaler.fit_transform(X_train)
X_train_trans = scaler.transform(X_train)

pca = PCA(n_components=20)
pca.fit(X_train_trans)
X_train_pca = pca.transform(X_train_trans)

clf = RandomForestClassifier(n_estimators=15)
clf.fit(X_train_pca, y_train)


X_test_trans = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_trans)
y_pred = clf.predict(X_test_pca)
print(accuracy_score(y_test, y_pred))