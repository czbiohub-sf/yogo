#! /usr/bin/bash
#
# usage: speedtest.py [-h] [--N N] [--BS BS] [--print-header PRINT_HEADER] pth_path
# speedtest.py: error: the following arguments are required: pth_path


rm speedtest_results.csv
for power in $(seq 0 10); do

  bs=$(echo "2^$power" | bc)

  if [ $bs -eq 1 ]
  then
    ./speedtest.py ../trained_models/bumbling-night-1847/best.pth --N 1000 --BS $bs --print-header >> speedtest_results.csv
  else
    ./speedtest.py ../trained_models/bumbling-night-1847/best.pth --N 1000 --BS $bs >> speedtest_results.csv
  fi
done
