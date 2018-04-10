import pandas as pd
import numpy as np
import time
from string import ascii_lowercase
import pickle

df = pd.read_excel("MergedData.xlsx")
df = df.replace({'Male':0}, regex=True)
df = df.replace({'Female':1}, regex=True)
df = df.dropna()

dfLetters = df[['Name', 'Gender']]

t0 = time.time()

for c1 in ascii_lowercase:
    t_char1 = time.time()
    print(c1)

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
    print("Letter " + c1 + " took: " + str(time.time() - t_char1))

t1 = time.time()
print("Total runtime: " + str(t1-t0))

out = open("Merged3LetterScheme.pkl", "wb")
pickle.dump(dfLetters, out)
out.close()
