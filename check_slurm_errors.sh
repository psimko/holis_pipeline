#!/bin/bash

# Define error keywords (modify as needed)
error_keyword="Traceback"

# Get all output files (modify if needed)
output_files="slurm-*.out"

# Initialize counters
total_files=0
error_files=0

# Loop through output files
for file in $output_files; do
  total_files=$((total_files+1))

  # Check if file exists
  if [[ ! -f "$file" ]]; then
    continue
  fi

  # Search for error keywords in the file
  if grep "$error_keyword" "$file"; then
    error_files=$((error_files+1))
    echo "** Errors found in: $file"
  fi
done

# Summarize findings
echo ""
echo "Total output files: $total_files"
echo "Files with errors: $error_files"
