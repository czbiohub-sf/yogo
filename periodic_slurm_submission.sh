#! /usr/bin/sh

for i in $(seq $1); do
  echo "submitting $i / $1   ($(date))"
  sbatch sweep_launch.sh $2
  sleep 600
done
