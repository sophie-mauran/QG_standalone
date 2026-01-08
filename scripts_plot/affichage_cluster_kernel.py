import csv
import pandas
import matplotlib.pyplot as plt


rmse_kernel_seuil40_Rstate1 = pandas.read_csv('rmse_kernel_cluster_seuil40_Rstate1.csv')
rmse_kernel_seuil40_Rstate2 = pandas.read_csv('rmse_kernel_cluster_seuil40_Rstate2.csv')
rmse_kernel_seuil40_Rstate2_bis = pandas.read_csv('rmse_kernel_cluster_seuil40_Rstate2_bis.csv')
rmse_kernel_seuil30_Rstate2 = pandas.read_csv('rmse_kernel_cluster_seuil30_Rstate2.csv')



fig, (ax1,ax2) = plt.subplots(2,1)

# Comparaison avec et sans mini clusters
ax1.scatter(rmse_kernel_seuil40_Rstate2["cycle"], rmse_kernel_seuil40_Rstate2["RMSE_ap_assim"], s=15, marker ='o', label="RMSE avant élimination des petits clusters")
ax1.scatter(rmse_kernel_seuil40_Rstate2_bis["cycle"], rmse_kernel_seuil40_Rstate2_bis["RMSE_ap_assim"], s=15, marker ='v', label="RMSE après élimination des petits clusters")

ax2.scatter(rmse_kernel_seuil40_Rstate2["cycle"], rmse_kernel_seuil40_Rstate2["SPREAD_ap_assim"], s=30, marker ='+', color = "purple", label="SPREAD avant élimination des petits clusters")
ax2.scatter(rmse_kernel_seuil40_Rstate2_bis["cycle"], rmse_kernel_seuil40_Rstate2_bis["SPREAD_ap_assim"], s=15, marker ='x', label="SPREAD après élimination des petits clusters")
ax1.legend()
ax2.legend()
plt.title("Comparaison RMSE et SPREAD de l'ensemble sur 50 cycles d'assimilation avant et après élimination des petits clusters pour le noyau linéaire, loc par clusters, seuil = 40 et R_state = 2")

plt.show()



fig, (ax1,ax2) = plt.subplots(2,1)

# Comparaison avec et sans mini clusters
ax1.scatter(rmse_kernel_seuil40_Rstate2["cycle"], rmse_kernel_seuil40_Rstate2["RMSE_ap_assim"], s=15, marker ='o', label="RMSE pour R_state = 2")
ax1.scatter(rmse_kernel_seuil40_Rstate1["cycle"], rmse_kernel_seuil40_Rstate1["RMSE_ap_assim"], s=15, marker ='v', label="RMSE pour R_state = 1")

ax2.scatter(rmse_kernel_seuil40_Rstate2["cycle"], rmse_kernel_seuil40_Rstate2["SPREAD_ap_assim"], s=30, marker ='+', color = "purple", label="SPREAD pour R_state = 2")
ax2.scatter(rmse_kernel_seuil40_Rstate1["cycle"], rmse_kernel_seuil40_Rstate1["SPREAD_ap_assim"], s=15, marker ='x', label="SPREAD pour R_state = 1")
ax1.legend()
ax2.legend()
plt.title("Influence du paramètre R_state sur la RMSE et le SPREAD de l'ensemble sur 50 cycles pour le noyau linéaire, loc par clusters, seuil = 40 ")

plt.show()


fig, (ax1,ax2) = plt.subplots(2,1)

# Comparaison avec et sans mini clusters
ax1.scatter(rmse_kernel_seuil40_Rstate2["cycle"], rmse_kernel_seuil40_Rstate2["RMSE_ap_assim"], s=15, marker ='o', label="RMSE pour seuil = 40")
ax1.scatter(rmse_kernel_seuil30_Rstate2["cycle"], rmse_kernel_seuil30_Rstate2["RMSE_ap_assim"], s=15, marker ='v', label="RMSE pour seuil = 30")

ax2.scatter(rmse_kernel_seuil40_Rstate2["cycle"], rmse_kernel_seuil40_Rstate2["SPREAD_ap_assim"], s=30, marker ='+', color = "purple", label="SPREAD pour seuil = 40")
ax2.scatter(rmse_kernel_seuil30_Rstate2["cycle"], rmse_kernel_seuil30_Rstate2["SPREAD_ap_assim"], s=15, marker ='x', label="SPREAD pour seuil = 30")
ax1.legend()
ax2.legend()
plt.title("Influence du paramètre de seuil sur la RMSE et le SPREAD de l'ensemble sur 50 cycles pour le noyau linéaire, loc par clusters, R_state = 2 ")

plt.show()



