import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
import operator
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import time
from string import ascii_lowercase
import pickle
from scipy.sparse import csr_matrix
import re
from sklearn.model_selection import KFold

in_df = open("MergedCompact.pkl", "rb")
dfCompact = pickle.load(in_df)
in_df.close()

"""LOOCV ANALYSIS"""
# Change parameter depending on target feature
dfLooTest = dfCompact.drop(["Name", "dataID"], axis=1)

# Timing LOOCV runtime
t0 = time.time()

# Uses Leave One Out Cross Validation (LOOCV) to estimate how training model would do on predicting external data
loo = KFold(n_splits=10)
x = dfLooTest.drop(['Gender'], axis=1).as_matrix()
y = dfLooTest[['Gender']].as_matrix().ravel()

male_total = 0
correct_male = 0
female_total = 0
correct_female = 0

errNames = [('Gender', 'Name')]

# Can optionally recreate a model to exclude the last training round
model = LogisticRegression(C=1e5)

for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    if y_test == [0]:
        male_total += 1
        correct_male += 1 if prediction == y_test else 0
    else:
        female_total += 1
        correct_female += 1 if prediction == y_test else 0
    if y_test != prediction:
        errNames.append((y_test, dfLetters.Name[test_index]))
overallAcc = (correct_male + correct_female) / (male_total + female_total)
maleAcc = correct_male / male_total
femaleAcc = correct_female / female_total

t1 = time.time()
runtime = t1 - t0

coefficients = {}
for i in range(len(dfLooTest.columns) - 1):
    coefficients[dfLooTest.columns[i + 1]] = model.coef_[0][i]

sorted_x = sorted(coefficients.items(), key=operator.itemgetter(1), reverse=True)
print(dfLooTest.columns)
print("Runtime: " + str(runtime))
print("Overall accuracy: " + str(overallAcc))
print("Male accuracy: " + str(maleAcc))
print("Female accuracy: " + str(femaleAcc))
print("\nCoefficients: \n")
print({'Total male': male_total, 'Correct male': correct_male, 'Total female': female_total, 'Correct female': correct_female})


dfCoefficients = pd.DataFrame(sorted_x)

dfWrongNames = pd.DataFrame(errNames)

out_coef = open("compact_merge_coef.pkl", "wb")
pickle.dump(dfCoefficients, out_coef)
out_coef.close()

out_wrong = open("compact_merge_wrong_names.pkl", "wb")
pickle.dump(dfWrongNames, out_wrong)
out_wrong.close()
