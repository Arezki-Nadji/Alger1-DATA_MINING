from sklearn import cluster
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import numpy as np

points=pd.DataFrame({ 'x': [1.95, 1.62, 3.12,0.91, 2.37, 5.2, 5.74, 3, 4.7, 4.97], 'y': [0.97, 0.74, 1.85, 1.09, 4.11, 2.52, 5.04, 3.47, 3.65, 3.32]})

Z = linkage(points,method='single',metric='cityblock')

groupes_cah = fcluster(Z,t=1,criterion='distance')
print(groupes_cah)
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(points)
#index triés des groupes
idk = np.argsort(kmeans.labels_)
#affichage des observations et leurs groupes
groupe =pd.DataFrame(points.index[idk],kmeans.labels_[idk])
#distances aux centres de classes des observations
distanc_centre =kmeans.transform(points)
#correspondance avec les groupes de la CAH
corrs = pd.crosstab(groupes_cah,kmeans.labels_)

#librairie pour évaluation des partitions
from sklearn import metrics
#utilisation de la métrique "silhouette"
#faire varier le nombre de clusters de 2 à 10
res = np.arange(9,dtype="double")
for k in np.arange(2):
    km = cluster.KMeans(n_clusters=k+2)
    km.fit(points)
    res[k] = metrics.silhouette_score(points,km.labels_)
    
sil = res

#graphique
import matplotlib.pyplot as plt
plt.title("Silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,11,1),res)
plt.show()

#moyenne par variable
m = points.mean()
#TSS
TSS = points.shape[0]*points.var(ddof=0)
print(TSS)
#data.frame conditionnellement aux groupes
gb = points.groupby(kmeans.labels_)
#effectifs conditionnels
nk = gb.size()
print(nk)
#moyennes conditionnelles
mk = gb.mean()
print(mk)
#pour chaque groupe écart à la moyenne par variable
EMk = (mk-m)**2
#pondéré par les effectifs du groupe
EM = EMk.multiply(nk,axis=0)
#somme des valeurs => BSS
BSS = np.sum(EM,axis=0)
print(BSS)
#carré du rapport de corrélation
#variance expliquée par l'appartenance aux groupes
#pour chaque variable
R2 = BSS/TSS
print(R2)

#ACP
from sklearn.decomposition import PCA
acp = PCA(n_components=2).fit_transform(points)
#projeter dans le plan factoriel
#avec un code couleur différent selon le groupe
#remarquer le rôle de zip() dans la boucle
for couleur,k in zip(['red','blue','lawngreen','aqua'],[0,1,2,3]):
    plt.scatter(acp[kmeans.labels_==k,0],acp[kmeans.labels_==k,1],c=couleur)
plt.show() 



distortions = []
K = range(1,10)
for k in K:
    kmeanModel = cluster.KMeans(n_clusters=k)
    kmeanModel.fit(points)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Clus')
plt.ylabel('Distortion')
plt.title('ELBOW')
plt.show()