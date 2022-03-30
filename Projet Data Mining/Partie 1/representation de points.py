import pandas as pd
import matplotlib.pyplot as plt

points=pd.DataFrame({ 'x': [1.95, 1.62, 3.12,0.91, 2.37, 5.2, 5.74, 3, 4.7, 4.97],
                      'y': [0.97, 0.74, 1.85, 1.09, 4.11, 2.52, 5.04, 3.47, 3.65, 3.32]})

#affichage de la distribution de données
plt.scatter(points['x'],points['y'])
plt.xlabel('X')
plt.ylabel('Y')

plt.show()