import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt

path = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/evol_nb_clusters/"


# Min pour fct de distance

rmse_minR10_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR10_seuil20 = pandas.read_csv(path+'davies_bouldin_R10_nblimit30_seuil20.csv')
#rmse_minR10_seuil30 = pandas.read_csv(path+'davies_bouldin_R10_nblimit30_seuil30.csv')

rmse_minR9_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR9_seuil20 = pandas.read_csv(path+'davies_bouldin_R9_nblimit30_seuil20.csv')
#rmse_minR9_seuil30 = pandas.read_csv(path+'davies_bouldin_R9_nblimit30_seuil30.csv')

rmse_minR8_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR8_seuil20 = pandas.read_csv(path+'davies_bouldin_R8_nblimit30_seuil20.csv')
#rmse_minR8_seuil30 = pandas.read_csv(path+'davies_bouldin_R8_nblimit30_seuil30.csv')

rmse_minR7_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR7_seuil20 = pandas.read_csv(path+'davies_bouldin_R7_nblimit30_seuil20.csv')
#rmse_minR7_seuil30 = pandas.read_csv(path+'davies_bouldin_R7_nblimit30_seuil30.csv')

rmse_minR6_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR6_seuil20 = pandas.read_csv(path+'davies_bouldin_R6_nblimit30_seuil20.csv')
#rmse_minR6_seuil30 = pandas.read_csv(path+'davies_bouldin_R6_nblimit30_seuil30.csv')

rmse_minR5_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR5_seuil20 = pandas.read_csv(path+'davies_bouldin_R5_nblimit30_seuil20.csv')
#rmse_minR5_seuil30 = pandas.read_csv(path+'davies_bouldin_R5_nblimit30_seuil30.csv')

rmse_minR4_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR4_seuil20 = pandas.read_csv(path+'davies_bouldin_R4_nblimit30_seuil20.csv')
#rmse_minR4_seuil30 = pandas.read_csv(path+'davies_bouldin_R4_nblimit30_seuil30.csv')

rmse_minR3_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit30_seuil10_infl_1.02.csv')
#rmse_minR3_seuil20 = pandas.read_csv(path+'davies_bouldin_R3_nblimit30_seuil20.csv')
#rmse_minR3_seuil30 = pandas.read_csv(path+'davies_bouldin_R3_nblimit30_seuil30.csv')


db_cycles = rmse_minR10_seuil10["cycle"][100:]
db_clusters = rmse_minR10_seuil10["nb_clusters"][100:]
mean_db = np.mean(db_clusters)


plt.figure(figsize=(10, 6))
plt.scatter(db_cycles, db_clusters, s=10, marker = '+')
plt.ylim(top=10)
plt.axhline(y=mean_db, color='red', linestyle='--', linewidth=2)
plt.text(500, mean_db, f'Moyenne: {mean_db:.2f}', va='bottom', ha='right', color='red', fontsize=10)
plt.savefig(path+"nb_clusters_R10_seuil10_nblimit30.png")
