#!/bin/bash

rep_list=$(seq 1 1 5)

for rep in $rep_list; do
  clf_iter_num=30000
  filter_key="mimic_iv"
  data_fraction_list=$(seq 1 -0.2 0.2)
  for data_fraction in $data_fraction_list; do
    python full_info_train.py --clf_iter_num=$clf_iter_num --data_fraction=$data_fraction --digit=4 --filter_key=$filter_key
    python full_info_train.py --clf_iter_num=$clf_iter_num --data_fraction=$data_fraction --digit=3 --filter_key=$filter_key
  done
#  clf_iter_num=30000
#  filter_key="srrsh-hospitalization"
#  data_fraction_list=$(seq 1 -0.2 0.2)
#  for data_fraction in $data_fraction_list; do
#    python full_info_train.py --clf_iter_num=$clf_iter_num --data_fraction=$data_fraction --digit=4 --filter_key=$filter_key
#    python full_info_train.py --clf_iter_num=$clf_iter_num --data_fraction=$data_fraction --digit=3 --filter_key=$filter_key
#  done
#
#  clf_iter_num=200000
#  filter_key="srrsh-outpatient"
#  data_fraction_list=$(seq 1 -0.2 0.2)
#  for data_fraction in $data_fraction_list; do
#    python full_info_train.py --clf_iter_num="$clf_iter_num" --data_fraction="$data_fraction" --digit=4 --filter_key=$filter_key
#    python full_info_train.py --clf_iter_num="$clf_iter_num" --data_fraction="$data_fraction" --digit=3 --filter_key=$filter_key
  done
done
