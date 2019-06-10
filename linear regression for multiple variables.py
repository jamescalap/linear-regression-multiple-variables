from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from word2number import w2n

df = pd.read_csv(r"C:\Users\James Calap\Desktop\train.csv")
import math
median_test_score = math.floor(df['score'].mean())
df['score'] = df['score'].fillna(median_test_score)
df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)
model = linear_model.LinearRegression()
model.fit(df[['experience', 'score', 'interview']], df['salary'])
output = model.predict([[2, 9, 6]])
print(output