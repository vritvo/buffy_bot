#!/bin/bash

INPUT="$1"
OUTPUT="$2"

# Check if arguments are provided
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input.pdf> <output.pdf>"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found"
    exit 1
fi

# Convert PDF to individual pages
magick -density 200 "$INPUT" /tmp/page_%04d.png

# Process each page with random rotation
for page in /tmp/page_*.png; do
    # Generate random rotation between -0.9 and 0.9 using /dev/urandom
    rotation=$(awk -v seed="$RANDOM$RANDOM" 'BEGIN{srand(seed); printf "%.2f", -0.5 + rand()*1.8}')

    # Generate random noise between 0.5 and 1.5
    noise=$(awk -v seed="$RANDOM$RANDOM" 'BEGIN{srand(seed); printf "%.2f", 0.5 + rand()*1.0}')

    # Apply effects with random rotation
    magick "$page" \
        -rotate "$rotation" \
        -attenuate $noise +noise Multiplicative \
        -wave 2x1070 \
        -distort Barrel "0.0 0.0 -0.02" \
        -colorspace Gray \
        "$page"
done

# Combine back into PDF
magick /tmp/page_*.png "$OUTPUT"

# Cleanup
rm /tmp/page_*.png