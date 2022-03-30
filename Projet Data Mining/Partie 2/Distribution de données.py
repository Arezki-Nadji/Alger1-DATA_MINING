import pandas as pd
import matplotlib.pyplot as plt

covid=pd.read_excel("Dataset TP02 Clustering.xlsx")

#affichage de la distribution de données
ax = plt.axes(projection='3d')
ax.scatter(covid['#Cases'].tolist(), covid['#Recovered'].tolist(), covid['#Death'].tolist())
ax.set_xlabel('Cases')
ax.set_ylabel('Recovered')
ax.set_zlabel('Deaths');

plt.show()

