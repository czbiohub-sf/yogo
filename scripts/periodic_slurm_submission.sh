#! /bin/bash

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Error: Incorrect usage."
  echo "Usage: $0 [number of submissions] [sbatch argument] [sleep time in seconds (optional, default 500)]"
  exit 1
fi

if [ $# -eq 3 ]; then
  sleep_time=$3
else
  sleep_time=500
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for i in $(seq $1); do
  echo "submitting $i / $1   ($(date))"
  # sbatch "$SCRIPT_DIR/sweep_launch.sh" $2
  eval "sh \"$SCRIPT_DIR/sweep_launch.sh\" $2"
  if [ $i -ne $1 ]; then
    sleep $sleep_time
  fi
done
