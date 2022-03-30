import tkinter as tk
from tkinter import *
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import cluster
import numpy as np
from sklearn import metrics

#chargement de données dans un dataframe
covid=pd.read_excel("Dataset TP02 Clustering.xlsx")


#fonction qui affiche la distribution de donées-----------------------------------------------------------------------------------
def distribution():
    covid = pd.read_excel("Dataset TP02 Clustering.xlsx")

    ax = plt.axes(projection='3d')
    ax.scatter(covid['#Cases'].tolist(), covid['#Recovered'].tolist(), covid['#Death'].tolist())
    ax.set_xlabel('Cases')
    ax.set_ylabel('Recovered')
    ax.set_zlabel('Deaths');

    plt.show()

#fonction qui affiche le dendogramme du single linkage-----------------------------------------------------------------------------------
def single():
    # calcule la matrice de distances
    Z = linkage(covid[['#Cases', '#Recovered', '#Death']], method='single', metric='euclidean')

    # affichage du dendogramme
    plt.title("Single Linkage")

    dendrogram(Z, labels=covid['#City'].tolist())

    plt.show()

#fonction qui affiche le dendogramme du average linkage----------------------------------------------------------------------------------
def average():
    # calcule la matrice de distances
    Z = linkage(covid[['#Cases', '#Recovered', '#Death']], method='average', metric='euclidean')

    # affichage du dendogramme
    plt.title("Average Linkage")

    dendrogram(Z, labels=covid['#City'].tolist())

    plt.show()


#fonction qui affiche le dendogramme du complete linkage---------------------------------------------------------------------------------
def complete():
    # calcule la matrice de distances
    Z = linkage(covid[['#Cases', '#Recovered', '#Death']], method='complete', metric='euclidean')

    # affichage du dendogramme
    plt.title("Complete Linkage")

    dendrogram(Z, labels=covid['#City'].tolist())

    plt.show()

#fonction pour plotter le graphe silhouette
def silhouette():

    # utilisation de la métrique "silhouette"
    # faire varier le nombre de clusters de 2 à 10
    res = np.arange(9, dtype="double")
    for k in np.arange(2):
        km = cluster.KMeans(n_clusters=k + 2)
        km.fit(covid[['#Cases', '#Recovered', '#Death']])
        res[k] = metrics.silhouette_score(covid[['#Cases', '#Recovered', '#Death']], km.labels_)

    sil = res

    # graphique
    import matplotlib.pyplot as plt
    plt.title("Silhouette")
    plt.xlabel("# of clusters")
    plt.plot(np.arange(2, 11, 1), res)
    plt.show()

##fonction pour plotter le graphe elbow
def elbow():
    distortions = []
    K = range(1, 48)
    for k in K:
        kmeanModel = cluster.KMeans(n_clusters=k)
        kmeanModel.fit(covid[['#Cases', '#Recovered', '#Death']])
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Clus')
    plt.ylabel('Distortion')
    plt.title('ELBOW')
    plt.show()




#Affichage de la fenétre de l'application-----------------------------------------------------------------------------------------
root = tk.Tk()
root.title("Application")
frame = tk.Frame(root)
frame.pack()
root.geometry("500x400")

label=Label(frame,text="Veuillez cliquer sur la fonctionnalité dont vous avez besoin:")
label.grid(row=0,pady = 5)

label2=Label(frame,text="Distribution de données:")
label2.grid(row=1,pady = 5)

distribution=tk.Button(frame,  text="Graphe de Distribution",command=distribution, width=20, height=1)
distribution.grid(row=2,pady=5)

label2=Label(frame,text="Classification Hiéarchique Ascendante:")
label2.grid(row=3,pady = 5)


single = tk.Button(frame,  text="Single Linkage",command=single, width=20, height=1)
single.grid(row=4,pady = 5)

average = tk.Button(frame, text="Average Linkage", width=20, height=1, command=average)
average.grid(row=5,pady = 5)

complete = tk.Button(frame, text="Complete Linkage", width=20, height=1, command=complete)
complete.grid(row=6,pady = 5)

label2=Label(frame,text="Kmeans:")
label2.grid(row=7,pady = 5)

silhouette = tk.Button(frame, text="Graphe de silhouette", width=20, height=1, command=silhouette)
silhouette.grid(row=8,pady = 5)

elbow = tk.Button(frame, text="Graphe d'Elbow", width=20, height=1, command=elbow)
elbow.grid(row=9,pady = 5)

root.mainloop()