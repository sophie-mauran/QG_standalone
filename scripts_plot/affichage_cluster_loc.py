import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt

path = "/home/smauran/Developpement/QG_standalone_counillon_assim4pdt/resultats_rmse/"

rmse_LETKF = pandas.read_csv(path+'rmse_LETKF_Sakov_R10_infl_102_2.csv')


''' # Min pour fct de distance
rmse_minR10_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil10_infl_1.03.csv')
rmse_minR10_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil20_infl_1.04.csv')
rmse_minR10_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil30_infl_1.05.csv')
#rmse_minR10_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil40.csv')

rmse_minR9_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil10_infl_1.03.csv')
rmse_minR9_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil20_infl_1.04.csv')
rmse_minR9_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil30_infl_1.05.csv')
#rmse_minR9_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil40.csv')

rmse_minR8_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil10_infl_1.03.csv')
rmse_minR8_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil20_infl_1.04.csv')
rmse_minR8_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR8_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil40.csv')

rmse_minR7_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil10_infl_1.03.csv')
rmse_minR7_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil20_infl_1.04.csv')
rmse_minR7_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR7_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil40.csv')

rmse_minR6_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil10_infl_1.03.csv')
rmse_minR6_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil20_infl_1.03.csv')
rmse_minR6_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR6_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil40.csv')

rmse_minR5_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil10_infl_1.02_3.csv')
rmse_minR5_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil20_infl_1.03.csv')
rmse_minR5_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR5_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil40.csv')

rmse_minR4_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil10_infl_1.02.csv')
rmse_minR4_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil20_infl_1.03.csv')
rmse_minR4_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR4_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil40.csv')

rmse_minR3_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil10_infl_1.02.csv')
rmse_minR3_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil20_infl_1.03.csv')
rmse_minR3_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil30_infl_1.03.csv')
#rmse_minR3_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil40.csv') '''


 # Min pour fct de distance
rmse_minR10_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil10_infl_1.03.csv')
rmse_minR10_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil20_infl_1.04.csv')
rmse_minR10_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR10_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R10_nblimit5_seuil40.csv')

rmse_minR9_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil10_infl_1.03.csv')
rmse_minR9_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil20_infl_1.04.csv')
rmse_minR9_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR9_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R9_nblimit5_seuil40.csv')

rmse_minR8_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil10_infl_1.03.csv')
rmse_minR8_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil20_infl_1.04.csv')
rmse_minR8_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR8_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R8_nblimit5_seuil40.csv')

rmse_minR7_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil10_infl_1.03.csv')
rmse_minR7_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil20_infl_1.04.csv')
rmse_minR7_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR7_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R7_nblimit5_seuil40.csv')

rmse_minR6_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil10_infl_1.03.csv')
rmse_minR6_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil20_infl_1.04.csv')
rmse_minR6_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR6_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R6_nblimit5_seuil40.csv')

rmse_minR5_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil10_infl_1.03.csv')
rmse_minR5_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil20_infl_1.04.csv')
rmse_minR5_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR5_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R5_nblimit5_seuil40.csv')

rmse_minR4_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil10_infl_1.03.csv')
rmse_minR4_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil20_infl_1.04.csv')
rmse_minR4_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR4_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R4_nblimit5_seuil40.csv')

rmse_minR3_seuil10 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil10_infl_1.03.csv')
rmse_minR3_seuil20 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil20_infl_1.04.csv')
rmse_minR3_seuil30 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil30_infl_1.04.csv')
#rmse_minR3_seuil40 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil40.csv')

#rmse_minR3_seuil20_infl1_01 = pandas.read_csv(path+'rmse_cluster_min_R3_nblimit5_seuil20_infl_1_03.csv')


# Mean pour fct de distance
rmse_meanR6_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R6_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR6_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R6_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR6_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R6_nblimit5_seuil10_infl_1.04.csv')

rmse_meanR7_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R7_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR7_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R7_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR7_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R7_nblimit5_seuil10_infl_1.04.csv')

rmse_meanR8_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R8_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR8_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R8_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR8_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R8_nblimit5_seuil10_infl_1.04.csv')

rmse_meanR9_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R9_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR9_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R9_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR9_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R9_nblimit5_seuil10_infl_1.04.csv')

rmse_meanR10_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R10_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR10_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R10_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR10_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R10_nblimit5_seuil10_infl_1.04.csv')

rmse_meanR11_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R11_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR11_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R11_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR11_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R11_nblimit5_seuil10_infl_1.04.csv')

rmse_meanR12_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R12_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR12_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R12_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR12_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R12_nblimit5_seuil10_infl_1.04.csv')

rmse_meanR13_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R13_nblimit5_seuil20_infl_1.04.csv')
rmse_meanR13_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R13_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR13_seuil10 = pandas.read_csv(path+'rmse_cluster_mean_R13_nblimit5_seuil10_infl_1.04.csv') 


rmse_meanR18_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R18_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR18_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R18_nblimit5_seuil20_infl_1.04.csv')


rmse_meanR23_seuil30 = pandas.read_csv(path+'rmse_cluster_mean_R23_nblimit5_seuil30_infl_1.04.csv')
rmse_meanR23_seuil20 = pandas.read_csv(path+'rmse_cluster_mean_R23_nblimit5_seuil20_infl_1.04.csv')

''' spread_av_min_R10_seuil20 = rmse_minR10_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_min_R9_seuil20 = rmse_minR9_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_min_R8_seuil20 = rmse_minR8_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_min_R7_seuil20 = rmse_minR7_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_min_R6_seuil20 = rmse_minR6_seuil20['SPREAD_av_assim'][150:501:2]
spread_av_min_R5_seuil20 = rmse_minR5_seuil20['SPREAD_av_assim'][150:501:2]
spread_av_min_R4_seuil20 = rmse_minR4_seuil20['SPREAD_av_assim'][150:501:2]
spread_av_min_R3_seuil20 = rmse_minR3_seuil20['SPREAD_av_assim'][150:501:2]


rmse_av_min_R10_seuil20 = rmse_minR10_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_min_R9_seuil20 = rmse_minR9_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_min_R8_seuil20 = rmse_minR8_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_min_R7_seuil20 = rmse_minR7_seuil20['RMSE_av_assim'][150:1001:2] 
rmse_av_min_R6_seuil20 = rmse_minR6_seuil20['RMSE_av_assim'][150:501:2]
rmse_av_min_R5_seuil20 = rmse_minR5_seuil20['RMSE_av_assim'][150:501:2]
rmse_av_min_R4_seuil20 = rmse_minR4_seuil20['RMSE_av_assim'][150:501:2]
rmse_av_min_R3_seuil20 = rmse_minR3_seuil20['RMSE_av_assim'][150:501:2]

spread_ap_min_R10_seuil20 = rmse_minR10_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_min_R9_seuil20 = rmse_minR9_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_min_R8_seuil20 = rmse_minR8_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_min_R7_seuil20 = rmse_minR7_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_min_R6_seuil20 = rmse_minR6_seuil20['SPREAD_av_assim'][150:501:2]
spread_ap_min_R5_seuil20 = rmse_minR5_seuil20['SPREAD_av_assim'][150:501:2]
spread_ap_min_R4_seuil20 = rmse_minR4_seuil20['SPREAD_av_assim'][150:501:2]
spread_ap_min_R3_seuil20 = rmse_minR3_seuil20['SPREAD_av_assim'][150:501:2]


rmse_ap_min_R10_seuil20 = rmse_minR10_seuil20['RMSE_av_assim'][150:501:2]
rmse_ap_min_R9_seuil20 = rmse_minR9_seuil20['RMSE_av_assim'][150:501:2]
rmse_ap_min_R8_seuil20 = rmse_minR8_seuil20['RMSE_av_assim'][150:501:2]
rmse_ap_min_R7_seuil20 = rmse_minR7_seuil20['RMSE_av_assim'][150:501:2]
rmse_ap_min_R6_seuil20 = rmse_minR6_seuil20['RMSE_av_assim'][150:501:2]
rmse_ap_min_R5_seuil20 = rmse_minR5_seuil20['RMSE_av_assim'][150:501:2]
rmse_ap_min_R4_seuil20 = rmse_minR4_seuil20['RMSE_av_assim'][150:501:2]
rmse_ap_min_R3_seuil20 = rmse_minR3_seuil20['RMSE_av_assim'][150:501:2]'''



''' spread_av_mean_R20_seuil20 = rmse_meanR20_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_mean_R19_seuil20 = rmse_meanR19_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_mean_R18_seuil20 = rmse_meanR18_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_mean_R17_seuil20 = rmse_meanR17_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_mean_R16_seuil20 = rmse_meanR16_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_mean_R15_seuil20 = rmse_meanR15_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_mean_R14_seuil20 = rmse_meanR14_seuil20['SPREAD_av_assim'][150:1001:2]
spread_av_mean_R13_seuil20 = rmse_meanR13_seuil20['SPREAD_av_assim'][150:1001:2]


rmse_av_mean_R20_seuil20 = rmse_meanR20_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_mean_R19_seuil20 = rmse_meanR19_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_mean_R18_seuil20 = rmse_meanR18_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_mean_R17_seuil20 = rmse_meanR17_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_mean_R16_seuil20 = rmse_meanR16_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_mean_R15_seuil20 = rmse_meanR15_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_mean_R14_seuil20 = rmse_meanR14_seuil20['RMSE_av_assim'][150:1001:2]
rmse_av_mean_R13_seuil20 = rmse_meanR13_seuil20['RMSE_av_assim'][150:1001:2]


spread_ap_mean_R20_seuil20 = rmse_meanR20_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_mean_R19_seuil20 = rmse_meanR19_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_mean_R18_seuil20 = rmse_meanR18_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_mean_R17_seuil20 = rmse_meanR17_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_mean_R16_seuil20 = rmse_meanR16_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_mean_R15_seuil20 = rmse_meanR15_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_mean_R14_seuil20 = rmse_meanR14_seuil20['SPREAD_av_assim'][150:1001:2]
spread_ap_mean_R13_seuil20 = rmse_meanR13_seuil20['SPREAD_av_assim'][150:1001:2]


rmse_ap_mean_R20_seuil20 = rmse_meanR20_seuil20['RMSE_av_assim'][150:1001:2]
rmse_ap_mean_R19_seuil20 = rmse_meanR19_seuil20['RMSE_av_assim'][150:1001:2]
rmse_ap_mean_R18_seuil20 = rmse_meanR18_seuil20['RMSE_av_assim'][150:1001:2]
rmse_ap_mean_R17_seuil20 = rmse_meanR17_seuil20['RMSE_av_assim'][150:1001:2]
rmse_ap_mean_R16_seuil20 = rmse_meanR16_seuil20['RMSE_av_assim'][150:1001:2]
rmse_ap_mean_R15_seuil20 = rmse_meanR15_seuil20['RMSE_av_assim'][150:1001:2]
rmse_ap_mean_R14_seuil20 = rmse_meanR14_seuil20['RMSE_av_assim'][150:1001:2]
rmse_ap_mean_R13_seuil20 = rmse_meanR13_seuil20['RMSE_av_assim'][150:1001:2] '''






fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharey=True)

ax1.set_ylim(0.0025, 0.011)

#plt.scatter(rmse_minR10_seuil20["cycle"][150:1001:2], rmse_minR10_seuil20["RMSE_ap_assim"][150:1001:2], s=15, marker ='o', label="rmse_minR10_seuil20")
ax1.plot(rmse_minR10_seuil10["cycle"][150:301:2], rmse_minR10_seuil10["RMSE_ap_assim"][150:301:2], label="R = 10")
#ax1.plot(rmse_minR9_seuil10["cycle"][150:301:2], rmse_minR9_seuil10["RMSE_ap_assim"][150:301:2], label="R = 9")
ax1.plot(rmse_minR8_seuil10["cycle"][150:301:2], rmse_minR8_seuil10["RMSE_ap_assim"][150:301:2], label="R = 8")
#ax1.plot(rmse_minR7_seuil10["cycle"][150:301:2], rmse_minR7_seuil10["RMSE_ap_assim"][150:301:2], label="R = 7")
ax1.plot(rmse_minR6_seuil10["cycle"][150:301:2], rmse_minR6_seuil10["RMSE_ap_assim"][150:301:2], label="R = 6")
#ax1.plot(rmse_minR5_seuil10["cycle"][150:301:2], rmse_minR5_seuil10["RMSE_ap_assim"][150:301:2], label="R = 5")
ax1.plot(rmse_minR4_seuil10["cycle"][150:301:2], rmse_minR4_seuil10["RMSE_ap_assim"][150:301:2], label="R = 4")
#ax1.plot(rmse_minR3_seuil10["cycle"][150:301:2], rmse_minR3_seuil10["RMSE_ap_assim"][150:301:2], label="R = 3")

ax1.legend(loc='upper left')
ax1.set_title("seuil = 10") 

ax2.plot(rmse_minR10_seuil20["cycle"][150:301:2], rmse_minR10_seuil20["RMSE_ap_assim"][150:301:2], label="R = 10")
#ax2.plot(rmse_minR9_seuil20["cycle"][150:301:2], rmse_minR9_seuil20["RMSE_ap_assim"][150:301:2], label="R = 9")
ax2.plot(rmse_minR8_seuil20["cycle"][150:301:2], rmse_minR8_seuil20["RMSE_ap_assim"][150:301:2], label="R = 8")
#ax2.plot(rmse_minR7_seuil20["cycle"][150:301:2], rmse_minR7_seuil20["RMSE_ap_assim"][150:301:2], label="R=7")
ax2.plot(rmse_minR6_seuil20["cycle"][150:301:2], rmse_minR6_seuil20["RMSE_ap_assim"][150:301:2], label="R = 6")
#ax2.plot(rmse_minR5_seuil20["cycle"][150:301:2], rmse_minR5_seuil20["RMSE_ap_assim"][150:301:2], label="R = 5")
ax2.plot(rmse_minR4_seuil20["cycle"][150:301:2], rmse_minR4_seuil20["RMSE_ap_assim"][150:301:2], label="R = 4")
#ax2.plot(rmse_minR3_seuil20["cycle"][150:301:2], rmse_minR3_seuil20["RMSE_ap_assim"][150:301:2], label="R = 3")

#ax2.legend(loc='upper left')
ax2.set_title("seuil = 20") 
ax2.legend(loc='upper left')

#ax3.set_ylim([0.0025, 0.02])

ax3.plot(rmse_minR10_seuil30["cycle"][150:301:2], rmse_minR10_seuil30["RMSE_ap_assim"][150:301:2], label="R = 10")
#ax3.plot(rmse_minR9_seuil30["cycle"][150:301:2], rmse_minR9_seuil30["RMSE_ap_assim"][150:301:2], label="R = 9")
ax3.plot(rmse_minR8_seuil30["cycle"][150:301:2], rmse_minR8_seuil30["RMSE_ap_assim"][150:301:2], label="R = 8")
#ax3.plot(rmse_minR7_seuil30["cycle"][150:301:2], rmse_minR7_seuil30["RMSE_ap_assim"][150:301:2], label="R = 7")
ax3.plot(rmse_minR6_seuil30["cycle"][150:301:2], rmse_minR6_seuil30["RMSE_ap_assim"][150:301:2], label="R = 6")
#ax3.plot(rmse_minR5_seuil30["cycle"][150:301:2], rmse_minR5_seuil30["RMSE_ap_assim"][150:301:2], label="R = 5")
ax3.plot(rmse_minR4_seuil30["cycle"][150:301:2], rmse_minR4_seuil30["RMSE_ap_assim"][150:301:2], label="R = 4")
#ax3.plot(rmse_minR3_seuil30["cycle"][150:301:2], rmse_minR3_seuil30["RMSE_ap_assim"][150:301:2], label="R = 3")

#ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.set_title("seuil = 30")
ax3.legend(loc='upper left')
#fig.suptitle("RMSE de l'ensemble sur 900 cycles d'assimilation (après warm-up de 100 cycles) avec la localisation par clusters, en fonction du rayon de localisation et de la fonction de calcul des distances")


fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharey=True)

ax1.set_ylim(0.0025, 0.011)

#ax1.plot(rmse_meanR13_seuil10["cycle"][150:301:2], rmse_meanR13_seuil10["RMSE_ap_assim"][150:301:2], label="R = 13")
ax1.plot(rmse_meanR12_seuil10["cycle"][150:301:2], rmse_meanR12_seuil10["RMSE_ap_assim"][150:301:2], label="R = 12")
#ax1.plot(rmse_meanR11_seuil10["cycle"][150:301:2], rmse_meanR11_seuil10["RMSE_ap_assim"][150:301:2], label="R = 11")
ax1.plot(rmse_meanR10_seuil10["cycle"][150:301:2], rmse_meanR10_seuil10["RMSE_ap_assim"][150:301:2], label="R = 10")
#ax1.plot(rmse_meanR9_seuil10["cycle"][150:301:2], rmse_meanR9_seuil10["RMSE_ap_assim"][150:301:2], label="R = 9")
ax1.plot(rmse_meanR8_seuil10["cycle"][150:301:2], rmse_meanR8_seuil10["RMSE_ap_assim"][150:301:2], label="R = 8")
#ax1.plot(rmse_meanR7_seuil10["cycle"][150:301:2], rmse_meanR7_seuil10["RMSE_ap_assim"][150:301:2], label="R = 7")
ax1.plot(rmse_meanR6_seuil10["cycle"][150:301:2], rmse_meanR6_seuil10["RMSE_ap_assim"][150:301:2], label="R = 6")

ax1.legend(loc='upper left')
ax1.set_title("seuil = 10") 

#ax2.plot(rmse_meanR13_seuil20["cycle"][150:301:2], rmse_meanR13_seuil20["RMSE_ap_assim"][150:301:2], label="R = 13")
ax2.plot(rmse_meanR23_seuil20["cycle"][150:301:2], rmse_meanR23_seuil20["RMSE_ap_assim"][150:301:2], label="R = 23")
#ax2.plot(rmse_meanR11_seuil20["cycle"][150:301:2], rmse_meanR11_seuil20["RMSE_ap_assim"][150:301:2], label="R = 11")
ax2.plot(rmse_meanR18_seuil20["cycle"][150:301:2], rmse_meanR18_seuil20["RMSE_ap_assim"][150:301:2], label="R = 18")
#ax2.plot(rmse_meanR9_seuil20["cycle"][150:301:2], rmse_meanR9_seuil20["RMSE_ap_assim"][150:301:2], label="R = 9")
ax2.plot(rmse_meanR13_seuil20["cycle"][150:301:2], rmse_meanR13_seuil20["RMSE_ap_assim"][150:301:2], label="R = 13")
#ax2.plot(rmse_meanR7_seuil20["cycle"][150:301:2], rmse_meanR7_seuil20["RMSE_ap_assim"][150:301:2], label="R = 7")
ax2.plot(rmse_meanR10_seuil20["cycle"][150:301:2], rmse_meanR10_seuil20["RMSE_ap_assim"][150:301:2], label="R = 10")
#ax2.legend(loc='upper left')
ax2.set_title("seuil = 20") 
ax2.legend(loc='upper left')

#ax3.set_ylim([0.0025, 0.02])

#ax3.plot(rmse_meanR13_seuil30["cycle"][150:301:2], rmse_meanR13_seuil30["RMSE_ap_assim"][150:301:2], label="R = 13")
ax3.plot(rmse_meanR23_seuil30["cycle"][150:301:2], rmse_meanR23_seuil30["RMSE_ap_assim"][150:301:2], label="R = 23")
#ax3.plot(rmse_meanR11_seuil30["cycle"][150:301:2], rmse_meanR11_seuil30["RMSE_ap_assim"][150:301:2], label="R = 11")
ax3.plot(rmse_meanR18_seuil30["cycle"][150:301:2], rmse_meanR18_seuil30["RMSE_ap_assim"][150:301:2], label="R = 18")
#ax3.plot(rmse_meanR9_seuil30["cycle"][150:301:2], rmse_meanR9_seuil30["RMSE_ap_assim"][150:301:2], label="R = 9")
ax3.plot(rmse_meanR13_seuil30["cycle"][150:301:2], rmse_meanR13_seuil30["RMSE_ap_assim"][150:301:2], label="R = 13")
#ax3.plot(rmse_meanR7_seuil30["cycle"][150:301:2], rmse_meanR7_seuil30["RMSE_ap_assim"][150:301:2], label="R = 7")
ax3.plot(rmse_meanR10_seuil30["cycle"][150:301:2], rmse_meanR10_seuil30["RMSE_ap_assim"][150:301:2], label="R = 10")

#ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.set_title("seuil = 30")
ax3.legend(loc='upper left')
#fig.suptitle("RMSE de l'ensemble sur 900 cycles d'assimilation (après warm-up de 100 cycles) avec la localisation par clusters, en fonction du rayon de localisation et de la fonction de calcul des distances")




'''fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharey=True)


ax1.plot(rmse_minR10_seuil10["cycle"][150:501:2], rmse_minR10_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 10")
ax1.plot(rmse_minR9_seuil10["cycle"][150:501:2], rmse_minR9_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 9")
ax1.plot(rmse_minR8_seuil10["cycle"][150:501:2], rmse_minR8_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 8")
ax1.plot(rmse_minR7_seuil10["cycle"][150:501:2], rmse_minR7_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 7")
ax1.plot(rmse_minR6_seuil10["cycle"][150:501:2], rmse_minR6_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 6")
ax1.plot(rmse_minR5_seuil10["cycle"][150:501:2], rmse_minR5_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 5")
ax1.plot(rmse_minR4_seuil10["cycle"][150:501:2], rmse_minR4_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 4")
ax1.plot(rmse_minR3_seuil10["cycle"][150:501:2], rmse_minR3_seuil10["SPREAD_ap_assim"][150:501:2], label="R = 3")

ax1.legend(loc='upper left')
ax1.set_title("seuil = 10")

#ax1.set_ylim([0.003, 0.017])
ax2.plot(rmse_minR10_seuil20["cycle"][150:1001:2], rmse_minR10_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 10")
ax2.plot(rmse_minR9_seuil20["cycle"][150:1001:2], rmse_minR9_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 9")
ax2.plot(rmse_minR8_seuil20["cycle"][150:1001:2], rmse_minR8_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 8")
ax2.plot(rmse_minR7_seuil20["cycle"][150:1001:2], rmse_minR7_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 7")
ax2.plot(rmse_minR6_seuil20["cycle"][150:501:2], rmse_minR6_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 6")
ax2.plot(rmse_minR5_seuil20["cycle"][150:501:2], rmse_minR5_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 5")
ax2.plot(rmse_minR4_seuil20["cycle"][150:501:2], rmse_minR4_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 4")
ax2.plot(rmse_minR3_seuil20["cycle"][150:501:2], rmse_minR3_seuil20["SPREAD_ap_assim"][150:501:2], label="R = 3")

#ax2.legend(loc='upper left')
ax2.set_title("seuil = 20")


#ax2.set_ylim([0.003, 0.017])

ax3.plot(rmse_minR10_seuil30["cycle"][150:501:2], rmse_minR10_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 10")
ax3.plot(rmse_minR9_seuil30["cycle"][150:501:2], rmse_minR9_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 9")
ax3.plot(rmse_minR8_seuil30["cycle"][150:501:2], rmse_minR8_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 8")
ax3.plot(rmse_minR7_seuil30["cycle"][150:501:2], rmse_minR7_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 7")
ax3.plot(rmse_minR6_seuil30["cycle"][150:501:2], rmse_minR6_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 6")
ax3.plot(rmse_minR5_seuil30["cycle"][150:501:2], rmse_minR5_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 5")
ax3.plot(rmse_minR4_seuil30["cycle"][150:501:2], rmse_minR4_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 4")
ax3.plot(rmse_minR3_seuil30["cycle"][150:501:2], rmse_minR3_seuil30["SPREAD_ap_assim"][150:501:2], label="R = 3")


#ax2.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5))
ax3.set_title("seuil = 30")
plt.show()'''


#fig.suptitle("SPREAD de l'ensemble sur 900 cycles d'assimilation (après warm-up de 100 cycles) avec la localisation par clusters, en fonction du rayon de localisation et de la fonction de calcul des distances")




fig, ax = plt.subplots()

#plt.plot(rmse_LETKF["cycle"][150::2], rmse_LETKF["RMSE_ap_assim"][150::2], label ="RMSE LETKF classique")
#plt.plot(rmse_LETKF["cycle"][150::2], rmse_LETKF["SPREAD_ap_assim"][150::2], label ="Spread LETKF classique")
#plt.plot(rmse_meanR17_seuil20["cycle"][150::2], rmse_meanR17_seuil20["RMSE_ap_assim"][100::2], label="rmse_meanR17_seuil20")
#plt.plot(rmse_meanR17_seuil30["cycle"][150::2], rmse_meanR17_seuil30["RMSE_ap_assim"][100::2], label="rmse_meanR17_seuil30")
#plt.plot(rmse_meanR17_seuil40["cycle"][150::2], rmse_meanR17_seuil40["RMSE_ap_assim"][100::2], label="rmse_meanR17_seuil40")
#plt.plot(rmse_minR10_seuil30["cycle"][150::2], rmse_minR10_seuil30["RMSE_ap_assim"][150::2], label="RMSE LETKF clustering")
#plt.plot(rmse_minR10_seuil30["cycle"][150::2], rmse_minR10_seuil30["SPREAD_ap_assim"][150::2], label="Spread LETKF clustering")
#plt.plot(rmse_minR4_seuil30["cycle"][100::2], rmse_minR4_seuil30["RMSE_ap_assim"][100::2], label="rmse_minR4_seuil30")
#plt.plot(rmse_meanR13_seuil20["cycle"][100::2], rmse_meanR13_seuil20["RMSE_ap_assim"][100::2], label="rmse_meanR13_seuil20")
#plt.plot(rmse_meanR14_seuil30["cycle"][100::2], rmse_meanR14_seuil30["RMSE_ap_assim"][100::2], label="rmse_meanR14_seuil30")
#plt.plot(rmse_minR3_seuil40["cycle"][100::2], rmse_minR3_seuil40["RMSE_ap_assim"][100::2], label="rmse_minR3_seuil40")
#plt.plot(rmse_meanR13_seuil40["cycle"][100::2], rmse_meanR13_seuil40["RMSE_ap_assim"][100::2], label="rmse_meanR13_seuil40")

ax.legend()
#plt.title("RMSE de l'ensemble sur 200 cycles d'assimilation (après warm-up de 100 cycles) avec la localisation par clusters, seuil = 30, en fonction du rayon de localisation, fonction de distance MIN")

#plt.show()



''' path = "/home/smauran/Developpement/QG_standalone_counillon/analyse_R4_seuil20/"
srf = pandas.read_csv(path+'srf_global.csv')
srf = srf['SRF_global'][:]
rmse_ap_min_R4_seuil20 = rmse_minR4_seuil20['RMSE_av_assim'][105:611:5]
cycles = rmse_minR4_seuil20['cycle'][105:611:5]


fig, ax1 = plt.subplots(figsize=(10, 6))


ax1.plot(cycles, srf, label ="SRF global")
ax1.set_ylabel('SRF', color='b')
ax1.tick_params(axis='y', labelcolor='b')


ax2 = ax1.twinx()
ax2.plot(cycles, rmse_ap_min_R4_seuil20, label="RMSE", color='r')
ax2.set_ylabel('RMSE', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

fig.tight_layout() 
plt.title("RMSE et Spread Reduction Factor (SRF) de l'ensemble")'''

plt.show()
