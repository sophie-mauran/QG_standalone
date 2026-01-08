#!/bin/sh


## Don't forget to delete/rename members from a previous generation before running the script. 


# Ensemble size
N=$1
t_burn_in=$2

# simulation parameters file
#f_params_members="prms_sakov2008_ens.txt"
#f_params_truth="prms_sakov2008_truth.txt"

f_params_members="prms_counillon2009_ens.txt"
f_params_truth="prms_counillon2009_truth.txt"
dtout="$(grep -E "dtout" "truth/$f_params_members")"
dtout="${dtout##*=}"
tgen=$dtout
f_ens_start="/home/smauran/dpr_data/samples/QG_samples.npz"
rstart=$(echo "$tgen/$dtout+1" | bc)


# We modify the simulation end time to include the generation instant
chmod +x ./modif_fparams.sh
./modif_fparams.sh "truth/$f_params_members" "tend" "$tgen"
./modif_fparams.sh "truth/$f_params_members" "restartfname" "\"\""
./modif_fparams.sh "truth/$f_params_members" "rstart" "0"
./modif_fparams.sh "truth/$f_params_members" "outfname" "\"start_run.nc\""

cd truth
# Run the model until time t_gen
./qg "$f_params_members" 
cd ..



for i in `seq 1 $N` 
do
  # Then copy the truth as many times as there are members in the set. 
  cp -r truth member_$i
  # cd member_$i
  # # And we add noise to generate the member
  # # Then let run for a while to get away from the initial conditions.
  # ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "tend" "$tindep"
  # ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "restartfname" "\"start_run.nc\""
  # ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "rstart" "$rstart"
  # ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "outfname" "\"run_t_$tindep.nc\""
  # ./qg "$f_params_members" &
  
  # cd ..
done
python ~/Developpement/QG_standalone_counillon_assim4pdt/recup_ensemble.py $f_ens_start $N
echo "replacement done"

cd ~/Developpement/QG_standalone_counillon_assim4pdt/member_1
# Then let run for a while to get away from the initial conditions.
~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "tend" "$t_burn_in"
~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "restartfname" "\"start_run.nc\""
~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "rstart" "$rstart"
~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_truth" "outfname" "\"run_t_$t_burn_in.nc\""
./qg "$f_params_truth" &

# cd ..
echo "truth generated"


for i in `seq 2 $N`
do
  cd ~/Developpement/QG_standalone_counillon_assim4pdt/member_$i
  # Then let run for a while to get away from the initial conditions.
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "tend" "$t_burn_in"
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "restartfname" "\"start_run.nc\""
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "rstart" "$rstart"
  ~/Developpement/QG_standalone_counillon_assim4pdt/modif_fparams.sh "$f_params_members" "outfname" "\"run_t_$t_burn_in.nc\""
  ./qg "$f_params_members" &
  
  # cd ..
  echo "member_$i generated"
done
wait




