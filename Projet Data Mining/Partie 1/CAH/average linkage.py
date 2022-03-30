
#importation des données
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
points=pd.DataFrame({ 'x': [1.95, 1.62, 3.12,0.91, 2.37, 5.2, 5.74, 3, 4.7, 4.97],
            'y': [0.97, 0.74, 1.85, 1.09, 4.11, 2.52, 5.04, 3.47, 3.65, 3.32]})

#average linkage
Z = linkage(points,method='average',metric='euclidean')

#affichage du dendogramme
plt.title("Average Linkage")

dendrogram(Z,labels=["A","B","C","D","E","F","G","H","I","K"])

plt.show()

#Decoupage des groupes a partit du niveau 2.5
groupes_cah = fcluster(Z,t=2.5,criterion='distance')

#affichage des points et leurs groupes
rze=pd.DataFrame(["A","B","C","D","E","F","G","H","I","K"],groupes_cah)
print(rze)