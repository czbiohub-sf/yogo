#!/usr/bin/env bash

# This is from LabelStudio, with some minor modifications


INPUT_DIR=$1
PORT=${2:-8081}

echo "Scanning ${INPUT_DIR} ..."

echo "serving files from ${INPUT_DIR} to http://localhost:${PORT} ..."

green=`tput setaf 2`
reset=`tput sgr0`

if [ ! -d "$INPUT_DIR" ]; then
  echo "couldn't find $INPUT_DIR; double check that it exists, and that the path is correct"
  exit 1
fi

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
# does this have to be INPUT_DIR__ESCAPED? where
# INPUT_DIR_ESCAPED=$(printf '%s\n' "$INPUT_DIR" | sed -e 's/[\/&]/\\&/g')
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$INPUT_DIR"

echo "Running web server on the port ${PORT}"
cd "$INPUT_DIR" && python3 -m http.server $PORT
