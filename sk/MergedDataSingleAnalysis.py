import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
import operator
from openpyxl import load_workbook
from string import ascii_uppercase
from string import ascii_lowercase
import time

print("Libraries imported")

# ============== MergedData Analyses ================
# Has not been tested for bugs
# Check runtime

t0 = time.time()
df = pd.read_excel("MergedData.xlsx")
t1 = time.time()
print("MergedData read and took: " + str(t1 - t0))

dropList = ["Name", "dataID", "Last Letter", "Last Two"]

""" SINGLE FEATURE ANALYSIS """
# Timing LOOCV runtime
t0 = time.time()

singleFeatureAcc = [('Feature', 'Overall Accuracy', 'Male Accuracy', 'Female Accuracy')]

print("Analysis about to begin on " + str(len(df.columns) - len(dropList)) + "features")
count = 0
for currFeature in df:
    if currFeature in dropList or currFeature == 'Gender':
        continue
    dfLooTest = df[['Gender', currFeature]]
    print("Current feature: " + currFeature)

    # Uses Leave One Out Cross Validation (LOOCV) to estimate how training model would do on predicting external data
    loo = LeaveOneOut()
    x = dfLooTest.drop(['Gender'], axis=1).as_matrix()
    y = dfLooTest[['Gender']].as_matrix().ravel()

    male_total = 0
    correct_male = 0
    female_total = 0
    correct_female = 0

    # Can optionally recreate a model to exclude the last training round
    model = LogisticRegression(C=1e5)

    flag = True

    for train_index, test_index in loo.split(x):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        if flag:
            print("First prediction of: " + currFeature)
            print("Prediction: " + str(prediction))
            print(x_train, x_test, y_train, y_test)
        if y_test == [0]:
            male_total += 1
            correct_male += 1 if prediction == y_test else 0
        else:
            female_total += 1
            correct_female += 1 if prediction == y_test else 0
        flag = False
    overallAcc = (correct_male + correct_female) / (male_total + female_total)
    maleAcc = correct_male / male_total
    femaleAcc = correct_female / female_total

    print("Overall accuracy of " + currFeature + ": " + str(overallAcc))
    print("Feature number: " + str(count))
    count += 1

    singleFeatureAcc.append((currFeature, overallAcc, maleAcc, femaleAcc))

t1 = time.time()
singleFeatureRunTime = t1 - t0
print("Runtime of compact model single feature analysis: " + str(singleFeatureRunTime))

dfWrite = pd.DataFrame(singleFeatureAcc)

excel_dir = 'MergedDataSingleAnalysis.xlsx'
book = load_workbook(excel_dir)
with pd.ExcelWriter(excel_dir, engine='openpyxl') as writer:
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)    

    ## Your dataframe to append. 
    dfWrite.to_excel(writer, 'Sheet1', index=False)

    writer.save()  
writer.close()

