#importation des données
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
points=pd.DataFrame({ 'x': [1.95, 1.62, 3.12,0.91, 2.37, 5.2, 5.74, 3, 4.7, 4.97], 'y': [0.97, 0.74, 1.85, 1.09, 4.11, 2.52, 5.04, 3.47, 3.65, 3.32]})
points.set_index('x')
Z = linkage(points,method='average',metric='euclidean')


plt.title("CAH")

dendrogram(Z,labels=["A","B","C","D","E","F","G","H","I","K"])

plt.show()

groupes_cah = fcluster(Z,t=1,criterion='distance')
print(groupes_cah)

#index triés des groupes
import numpy as np
idg = np.argsort(groupes_cah)
#affichage des observations et leurs groupes
rze=pd.DataFrame(["A","B","C","D","E","F","G","H","I","K"],groupes_cah)
