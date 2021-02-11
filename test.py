import pickle
import pandas as pd
from sklearn import metrics

df = pd.read_csv("heart_attack_prediction_datset.csv")
feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[feature_columns]
Y = df.target

classifier = pickle.load(open('trained_model.pkl', 'rb'))
Y_pred = classifier.predict(X)
print("Accuracy:", metrics.accuracy_score(Y, Y_pred))

age = int(input("Enter age of the patient (Values -- 29 to 77) --> "))
sex = int(input("Enter '1' for MALE and '0' for FEMALE --> "))
cp = int(input("Enter chest pain type i.e., Type-1 to Type-4 --> "))-1
trestbps = int(input("Enter the resting blood pressure (in mm Hg on admission to the hospital) (Values -- 94 to 200) --> "))
chol = int(input("Enter the serum cholesterol in mg/dl (Values -- 126 to 564) --> "))
fbs = int(input("Enter '1' for TRUE and '0' for FALSE (fasting blood sugar & gt; 120 mg/dl) --> "))
restecg = int(input("Enter resting electro-cardiographic results (Values -- 0, 1, 2) --> "))
thalach = int(input("Enter maximum heart rate achieved (Values -- 71 to 202) --> "))
exang = int(input("Confirm whether the person has exercise induced angina ('1' for YES and '0' for NO) --> "))
oldpeak = float(input("Amount of ST depression induced by exercise relative to rest (Values -- 0 to 6.2) --> "))
slope = int(input("Enter the slope of the peak exercise ST segment (Values -- 0, 1, 2) --> "))
ca = int(input("Enter the number of major vessels colored by fluoroscopy (Values -- 0, 1, 2, 3) --> "))
thal = int(input("Enter whether the patients heart beat is normal (Values -- '0' for normal '1' for fixed defect '2' for reversable defect) --> "))

dataframe = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
pred = classifier.predict(dataframe)

if pred[0] == 0:
    print("The patient does not have a heart disease !!")
else:
    print("The patient has a heart disease !!")