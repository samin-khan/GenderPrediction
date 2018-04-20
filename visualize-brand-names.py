import pandas as pd
import pickle as pk
import scipy
import numpy as np
import scipy.stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

brandlist = ['nike','adidas','ford','lexus','olay','aveda','pandora','tylenol','gerber','piaget','swatch','tiffany']

fig = plt.figure(figsize=(12,6))

# Load HBC association matrix and sense indices
df = pd.read_csv('brand_predictions_features.csv' )
brand = list(df.BRAND)
cat = list(df.CATEGORY)
gender = np.asarray(df.PREDICTION)
featMatrix = np.asmatrix(df.iloc[:,3:20])

D = TSNE(n_components=2).fit_transform(featMatrix)
 
#plt.axvline(x=EC,color='black')
indF = np.where(gender==1)[0]
indM = np.where(gender==0)[0]

plt.scatter(D[indF,0],D[indF,1],s=1,color='red')
plt.scatter(D[indM,0],D[indM,1],s=1,color='blue')


for i in range(0,len(brandlist)):
    ind = brand.index(brandlist[i])
    if gender[ind] == 1:
        plt.text(D[ind,0],D[ind,1],brandlist[i],color='red',fontsize=12, fontweight='bold')
    else:
        plt.text(D[ind,0],D[ind,1],brandlist[i],color='blue',fontsize=12, fontweight='bold')
    
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
 
plt.legend(['Female','Male'])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Gender of brand names')
#plt.axis('off')
            
#plt.savefig('brandgender.eps',dpi=600)

plt.show()