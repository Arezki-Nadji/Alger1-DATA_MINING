import pandas as pd
from sklearn import cluster
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import numpy as np

covid = pd.read_excel("Dataset TP02 Clustering.xlsx")
Z = linkage(covid[['#Cases', '#Recovered', '#Death']],method='single',metric='euclidean')

groupes_cah = fcluster(Z,t=1,criterion='distance')



#Matrice de distance*******************
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(covid[['#Cases', '#Recovered', '#Death']])
#index triés des groupes
idk = np.argsort(kmeans.labels_)
#affichage des observations et leurs groupes
groupe =pd.DataFrame(covid[['#Cases', '#Recovered', '#Death']].index[idk],kmeans.labels_[idk])
#distances aux centres de classes des observations
distanc_centre =kmeans.transform(covid[['#Cases', '#Recovered', '#Death']])
#correspondance avec les groupes de la CAH
corrs = pd.crosstab(groupes_cah,kmeans.labels_)


#SIlouhette*****************************************
#librairie pour évaluation des partitions
from sklearn import metrics
#utilisation de la métrique "silhouette"
#faire varier le nombre de clusters de 2 à 10
res = np.arange(9,dtype="double")
for k in np.arange(2):
    km = cluster.KMeans(n_clusters=k+2)
    km.fit(covid[['#Cases', '#Recovered', '#Death']])
    res[k] = metrics.silhouette_score(covid[['#Cases', '#Recovered', '#Death']],km.labels_)
    
sil = res

#graphique
import matplotlib.pyplot as plt
plt.title("Silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,11,1),res)
plt.show()

#ELBOW***********************************************
distortions = []
K = range(1,48)
for k in K:
    kmeanModel = cluster.KMeans(n_clusters=k)
    kmeanModel.fit(covid[['#Cases', '#Recovered', '#Death']])
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Clus')
plt.ylabel('Distortion')
plt.title('ELBOW')
plt.show()
