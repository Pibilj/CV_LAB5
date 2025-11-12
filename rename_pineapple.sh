#!/bin/bash

# Bash Script to Rename Pineapple Images
# ======================================
# This script renames all images in the pineapple folder to
# pineapple_1.jpg, pineapple_2.jpg, pineapple_3.jpg, etc.

#1.)Here we are just setting a variable to the path of IMAGE_DIR?
# Set the directory where the images are located
IMAGE_DIR="/home/jay/Documents/computer_vision/CV_LAB5/fruit_images/pineapple/whole pineapple white background"
    

# Initialize a counter starting at 1
counter=1

#2.) Any non listed file type would not be read here?
# Loop through all image files (jpg, jpeg, png) in the directory
# The * wildcard matches any filename
for file in "$IMAGE_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    
    # Check if the file actually exists (handles case where no files match)
    # -e checks if the file exists
    if [ -e "$file" ]; then
        
        # Extract the file extension (e.g., .jpg, .png)
        # ${file##*.} means: remove everything up to the last dot
        extension="${file##*.}"
        
        # Extract just the directory path
        # ${file%/*} means: remove everything after the last /
        directory="${file%/*}"
        
        # Create the new filename: pineapple_1.jpg, pineapple_2.jpg, etc.
        new_name="${directory}/pineapple_${counter}.${extension}"
        
        # Rename the file
        # mv = move/rename command
        mv "$file" "$new_name"
        
        # Print what we did (optional, for feedback)
        echo "Renamed: $(basename "$file") → $(basename "$new_name")"
        
        # Increment the counter for the next file
        ((counter++))
    fi
done

echo ""
echo "✅ Done! Renamed $((counter - 1)) files."
