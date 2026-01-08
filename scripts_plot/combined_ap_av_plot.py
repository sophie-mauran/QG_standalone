import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les fichiers CSV
chemin_fichier1 = '../rmse_cluster_min_15_05.csv'
chemin_fichier2 = '../rmse_cluster_mean_15_05.csv'

df1 = pd.read_csv(chemin_fichier1)
df2 = pd.read_csv(chemin_fichier2)

# Extraire les colonnes pertinentes
spread_av_assim_1 = df1['SPREAD_av_assim'][100:]
spread_ap_assim_1 = df1['SPREAD_ap_assim'][100:]
spread_av_assim_2 = df2['SPREAD_av_assim'][100:]
spread_ap_assim_2 = df2['SPREAD_ap_assim'][100:]


rmse_av_assim_1 = df1['RMSE_av_assim'][100:]
rmse_ap_assim_1 = df1['RMSE_ap_assim'][100:]
rmse_av_assim_2 = df2['RMSE_av_assim'][100:]
rmse_ap_assim_2 = df2['RMSE_ap_assim'][100:]



# Intercaler les données dans une seule liste
interleaved_spread_min = []

for av, ap in zip(spread_av_assim_1, spread_ap_assim_1):
    interleaved_spread_min.append(av)
    interleaved_spread_min.append(ap)

interleaved_spread_mean = []

for av, ap in zip(spread_av_assim_2, spread_ap_assim_2):
    interleaved_spread_mean.append(av)
    interleaved_spread_mean.append(ap)


interleaved_rmse_min = []

for av, ap in zip(rmse_av_assim_1, rmse_ap_assim_1):
    interleaved_rmse_min.append(av)
    interleaved_rmse_min.append(ap)

interleaved_rmse_mean = []

for av, ap in zip(rmse_av_assim_2, rmse_ap_assim_2):
    interleaved_rmse_mean.append(av)
    interleaved_rmse_mean.append(ap)


time_steps = np.arange(0, len(interleaved_spread_min) / 2, 0.5)

# Tracer les données intercalées
fig,(ax1,ax2) = plt.subplots(3,1,figsize=(12, 6))
ax1.plot(time_steps,interleaved_spread_min, label='calcul des distances au cluster par min')
ax1.plot(time_steps,interleaved_spread_mean, label='calcul des distances au cluster par mean')
ax1.set_xlabel('cyle d\'assimilation')
ax1.set_ylabel('Spread')
ax1.set_title('Evolution du spread sur 200 cycles après période de warmup de 100 cycles')
ax1.legend()

ax2.plot(time_steps,interleaved_rmse_min, label='calcul des distances au cluster par min')
ax2.plot(time_steps,interleaved_rmse_mean, label='calcul des distances au cluster par mean')
ax2.set_xlabel('cyle d\'assimilation')
ax2.set_ylabel('RMSE')
ax2.set_title('Evolution de la rmse sur 200 cycles après période de warmup de 100 cycles')
ax2.legend()

plt.show()