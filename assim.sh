#!/bin/sh


# Ensemble size
N=30
# State space size
Nx=16641
# Nb of assimilation cycles
n=1000
# number of assimilation cycles in burn-in
#n_burnin=32
# Generate ensemble ?
gen_ens=false
# # time at which noise is applied relative to truth
# tgen=1
# time at which the simulation is considered to be independent of the initial conditions
t_burn_in=400

scaler_string="NormalizerL2"

# simulation parameters file
#f_params_members="prms_sakov2008_ens.txt"
#f_params_truth="prms_sakov2008_truth.txt"

f_params_members="prms_counillon2009_ens.txt"
f_params_truth="prms_counillon2009_truth.txt"
dtout="$(grep -E "dtout" "truth/$f_params_truth")"
dtout="${dtout##*=}"
rstart_sim=$(echo "$t_burn_in/$dtout" | bc)
# Start file for the ensemble
f_ens_start="/home/smauran/dpr_data/samples/QG_samples.npz"

tprev=$t_burn_in




## Ensemble not generated anymore, we get it from a file
if $gen_ens 
then
chmod +x ./generation_ensemble.sh
./generation_ensemble.sh $N $t_burn_in
echo "ensemble has been generated"
fi
#python addNoise.py $N $Nx $t_burn_in


# # Model run
# tfinal=$(echo "$t_burn_in+$dtout*($n+$n_burnin)" | bc)
# cd truth
# ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "tend" "$tfinal"
# ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "outfname" "\"true_run_gen.nc\""
# ./qg "$f_params_members" 
# cd ..


# assimilation cycles
#n_total=$(echo "$n+$n_burnin" | bc)

for cycle in `seq 1 $n`
do
  python get_spread.py $N $Nx $tprev
  # Forecast step
  tend=$(echo "$t_burn_in+$dtout*$cycle" | bc)
  #start=$(date +%s)
  # Ensemble integration
  echo "beginning of forecast step"
  for member in `seq 2 $N`
  do
    cd member_$member
    # This should run each member of the ensemble in parallel on a different processor
    ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "tend" "$tend"
    ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "restartfname" "\"run_t_$tprev.nc\""
    ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "rstart" "$rstart_sim"
    ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "outfname" "\"run_t_$tend.nc\""
    ~/Developpement/QG_standalone_counillon_assim4pdt/member_$member/qg ~/Developpement/QG_standalone_counillon_assim4pdt/member_$member/$f_params_members &
    cd ..

  done
  # This should allow for the script to resume when all the members have been integrated
  wait
  python get_spread.py $N $Nx $tend

  echo "end of forecast step"
  #end=$(date +%s)
  #duration=$(( $end - $start ))
  #echo "$duration (s)"

  # Generate truth

  cd member_1
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "tend" "$tend"
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "restartfname" "\"run_t_$tprev.nc\""
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "rstart" "$rstart_sim"
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "outfname" "\"run_t_$tend.nc\""
  ~/Developpement/QG_standalone_counillon_assim4pdt/member_1/qg ~/Developpement/QG_standalone_counillon_assim4pdt/member_1/$f_params_truth
  cd ..
  
  # Assimilation step
  
  # Get the obs
  echo "getting obs"
  # obs_record=$(echo "$t_burn_in/$dtout+$cycle" | bc)
  obs_record=1
  # echo "obs_record"
  # echo $obs_record
  cd member_1
  echo 'o' | ncks -v t,q,psi -d record,$obs_record run_t_$tend.nc obs_record.nc
  cd ..
  echo "obs in obs_record.nc"
  echo "beginning of assimilation"
  # python assim_script.py /home/smauran/Developpement/QG_standalone_counillon_assim4pdt/truth/obs_record.nc $N $tend $dtout
  python assim_LETKF_cluster.py /home/smauran/Developpement/QG_standalone_counillon_assim4pdt/member_1/obs_record.nc $N $tend $dtout $cycle $scaler_string
  echo "end of assimilation"
  
  
  tprev=$tend
  rstart_sim=2
done
