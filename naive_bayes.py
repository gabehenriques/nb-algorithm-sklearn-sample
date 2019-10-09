from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd

wx = ['sunny','sunny','overcast','rainy','rainy','rainy','overcast','sunny','sunny','rainy','sunny','overcast','overcast','rainy']
temp = ['hot','hot','hot','mild','cool','cool','cool','mild','cool','mild','mild','mild','hot','mild']
tyre = ['slick','slick','inter','full wet','full wet','full wet','inter','slick','slick','full wet','slick','inter','inter', 'full wet']
result = ['win','lose','win','lose','lose','win','win','lose','lose','win','lose','lose','lose','lose']

"""Encode target labels with value between 0 and n_classes-1;"""
le = preprocessing.LabelEncoder()

"""Fit label encoder and return encoded labels;"""
wx_encoded = le.fit_transform(wx)
temp_encoded = le.fit_transform(temp)
tyre_encoded = le.fit_transform(tyre)
label = le.fit_transform(result)

"""Return an iterator of tuples based on the iterable object;"""
features = list(zip(wx_encoded, temp_encoded, tyre_encoded))

"""Gaussian Naive Bayes (GaussianNB)"""
classifier = GaussianNB()

df = pd.DataFrame({'weather': wx_encoded, 'temperature':temp_encoded, 'tyre':tyre_encoded, 'result':label})

"""Train classifier against our training set;"""
model = classifier.fit(features, label)

"""Predict the outcome;"""
predicted_outcome = model.predict([[0,1,1]])
probability = model.predict_proba([[0,1,1]])

print(df)
print("Predicted Outcome: ", predicted_outcome)
print('Predicted Probabilities: ', probability)
