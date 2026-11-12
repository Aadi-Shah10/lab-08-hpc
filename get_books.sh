#!/bin/bash
set -euo pipefail

# Usage: bash get_books.sh <bucket-name> <book-directory>
# Example: bash get_books.sh public-mst3k /scratch/$USER/books
if [ "$#" -ne 2 ]; then
    echo "Usage: bash get_books.sh <bucket-name> <book-directory>" >&2
    exit 1
fi

bucket="$1"
book_dir="$2"

# Create the book directory if it does not exist yet
mkdir -p "$book_dir"

# Copy every .txt file from the bucket, and nothing else
aws s3 cp "s3://$bucket/" "$book_dir" --recursive --exclude "*" --include "*.txt"

# Absolute path of the book directory, so find prints absolute paths
book_dir_abs="$(realpath "$book_dir")"

# Write the header, then one book file per line in sorted order
echo "File path" > book_list.txt
find "$book_dir_abs" -type f -name "*.txt" | LC_ALL=C sort >> book_list.txt

# Count the books listed (skip the header line)
count=$(tail -n +2 book_list.txt | wc -l)
echo "Listed $count books in book_list.txt"
