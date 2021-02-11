####################### Dataset Description #######################
'''
Age : age of the patient
Sex : 1--Male 0--Female
cp : Chest pain type
trestbps : resting blood pressure (in mm Hg on admission to the hospital)
chol : serum cholesterol in mg/dl
fbs : (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
restecg : resting electro-cardiographic results
thalach : maximum heart rate achieved
exang : exercise induced angina (1 = yes; 0 = no)
oldpeak : ST depression induced by exercise relative to rest
slope : the slope of the peak exercise ST segment
ca : number of major vessels (0-3) colored by flourosopy
thal : 0 = normal; 1 = fixed defect; 2 = reversable defect
target : final outcome
'''
####################### Dataset Description #######################

import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("heart_attack_prediction_datset.csv")
feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[feature_columns]
Y = df.target

classifier = DecisionTreeClassifier(criterion='entropy', random_state=1)
classifier.fit(X, Y)
print(classifier)

list_pickle = open('trained_model.pkl', 'wb')
pickle.dump(classifier, list_pickle)
list_pickle.close()