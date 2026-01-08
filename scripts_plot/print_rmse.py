import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV dans un DataFrame
fichier = 'rmse.csv'  # Remplacez 'votre_fichier.csv' par le chemin vers votre fichier CSV
data = pd.read_csv(fichier)

# Séparation des colonnes en listes
colonne0 = data.iloc[:, 0].tolist()
colonne1 = data.iloc[:, 1].tolist()
colonne2 = data.iloc[:, 2].tolist()
colonne3 = data.iloc[:, 3].tolist()
colonne4 = data.iloc[:, 4].tolist()

# Tracer les courbes alternées
plt.figure(figsize=(8, 6))

plt.plot(colonne0, colonne2, label='RMSE')
plt.plot(colonne0, colonne4, label='Spread')
#plt.plot(colonne3, colonne4, label='Spread')

plt.xlabel('cycle')
plt.title('RMSE et Spread')
plt.legend()
plt.grid(True)
plt.show()