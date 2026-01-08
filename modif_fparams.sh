#!/bin/sh

f_params="$1"
var_a_modif="$2"
nouvelle_valeur="$3"
ancienne_valeur="$(grep -E ${var_a_modif} $f_params)"
ancienne_valeur="${ancienne_valeur##*=}"
sed -i "s/${var_a_modif} =${ancienne_valeur}/${var_a_modif} =${nouvelle_valeur}/g" ${f_params}
