#!/usr/bin/env bash
# Deletes everything inside the "Logs" folder,
# plus all files named "model.ckpt" or "results.nc" throughout the project.

set -euo pipefail

# 1. Clear the Logs directory (only its contents)
if [ -d "Logs" ]; then
  echo "Cleaning contents of Logs/ directory..."
  rm -rf Logs/*
else
  echo "Warning: Logs/ directory does not exist."
fi

# 2. Remove all instances of model.ckpt and results.nc recursively
echo "Deleting all 'model.ckpt' and 'results.nc' files in the project..."
find . -type f \( -name "model.ckpt" -o -name "results.nc" \) -exec rm -f {} +
echo "Deletion complete."