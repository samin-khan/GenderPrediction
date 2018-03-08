import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
import operator
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import time
from string import ascii_lowercase

df = pd.read_excel("MergedData.xlsx")
df = df.replace({'Male':0}, regex=True)
df = df.replace({'Female':1}, regex=True)
df = df.dropna()

dfLetters = df[['Name', 'Gender']]

for c1 in ascii_lowercase:
    dfLetters.insert(len(dfLetters.columns), "Starts with: " + c1, 0)
    dfLetters.loc[(dfLetters['Name'].str.startswith(c1)), "Starts with: " + c1] = 1
    
    dfLetters.insert(len(dfLetters.columns), "Ends with: " + c1, 0)
    dfLetters.loc[(dfLetters['Name'].str.endswith(c1)), "Ends with: " + c1] = 1
    
    for c2 in ascii_lowercase:
        dfLetters.insert(len(dfLetters.columns), "Starts with: " + c1 + c2, 0)
        dfLetters.loc[(dfLetters['Name'].str.startswith(c1 + c2)), "Starts with: " + c1 + c2] = 1

        dfLetters.insert(len(dfLetters.columns), "Ends with: " + c1 + c2, 0)
        dfLetters.loc[(dfLetters['Name'].str.endswith(c1 + c2)), "Ends with: " + c1 + c2] = 1
        
        for c3 in ascii_lowercase:
            dfLetters.insert(len(dfLetters.columns), "Starts with: " + c1 + c2 + c3, 0)
            dfLetters.loc[(dfLetters['Name'].str.startswith(c1 + c2 + c3)), "Starts with: " + c1 + c2 + c3] = 1

            dfLetters.insert(len(dfLetters.columns), "Ends with: " + c1 + c2 + c3, 0)
            dfLetters.loc[(dfLetters['Name'].str.endswith(c1 + c2 + c3)), "Ends with: " + c1 + c2 + c3] = 1

excel_dir = 'MergedData.xlsx'
book = load_workbook(excel_dir)
with pd.ExcelWriter(excel_dir, engine='openpyxl') as writer:
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)    

    ## Your dataframe to append. 
    dfWrite.to_excel(writer, 'Sheet4', index=False)

    writer.save()  
writer.close()
