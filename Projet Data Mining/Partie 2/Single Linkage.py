
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#chargement de données
covid=pd.read_excel("Dataset TP02 Clustering.xlsx")

#calcule la matrice de distances
Z = linkage(covid[['#Cases', '#Recovered', '#Death']],method='single',metric='euclidean')

#affichage du dendogramme
plt.title("Single Linkage")

dendrogram(Z,labels=covid['#City'].tolist())

plt.show()