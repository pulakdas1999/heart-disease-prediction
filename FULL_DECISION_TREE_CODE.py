import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_attack_prediction_datset.csv")

feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                   'fbs', 'restecg', 'thalach', 'exang',
                   'oldpeak', 'slope', 'ca', 'thal']

X = df[feature_columns]
Y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, shuffle=True)

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))