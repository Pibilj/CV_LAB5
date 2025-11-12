#!/bin/bash

# ==============================================================================
# BASH SCRIPT TUTORIAL: Rename Images with Incrementing Numbers
# ==============================================================================
# 
# PURPOSE: Rename all images in a folder to have a prefix + incrementing number
# EXAMPLE: image1.jpg, photo.png → pineapple_1.jpg, pineapple_2.jpg
#
# HOW TO USE:
#   1. Make executable: chmod +x rename_images.sh
#   2. Run: ./rename_images.sh <folder_path> <prefix>
#   Example: ./rename_images.sh ./pineapple pineapple
#
# ==============================================================================

# LESSON 1: SCRIPT ARGUMENTS
# ---------------------------
# $1 = first argument (folder path)
# $2 = second argument (prefix for new names)
# $# = total number of arguments

# Check if user provided the required arguments
if [ $# -ne 2 ]; then
    echo "❌ Error: Wrong number of arguments!"
    echo ""
    echo "Usage: $0 <folder_path> <prefix>"
    echo "Example: $0 ./pineapple pineapple"
    exit 1
fi

# Store arguments in variables with descriptive names
FOLDER_PATH="$1"
PREFIX="$2"

# LESSON 2: FILE CHECKS
# ----------------------
# -d checks if path is a directory
# -e checks if path exists

if [ ! -e "$FOLDER_PATH" ]; then
    echo "❌ Error: '$FOLDER_PATH' does not exist!"
    exit 1
fi

if [ ! -d "$FOLDER_PATH" ]; then
    echo "❌ Error: '$FOLDER_PATH' is not a directory!"
    exit 1
fi

echo "📁 Folder: $FOLDER_PATH"
echo "🏷️  Prefix: $PREFIX"
echo ""

# LESSON 3: COUNTERS AND LOOPS
# -----------------------------
counter=1

# LESSON 4: FOR LOOPS WITH WILDCARDS
# -----------------------------------
# The for loop iterates over all matching files
# *.{jpg,jpeg,png} matches files ending in .jpg OR .jpeg OR .png
# Quotes around "$FOLDER_PATH" handle spaces in directory names

for file in "$FOLDER_PATH"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    
    # LESSON 5: CONDITIONAL STATEMENTS
    # --------------------------------
    # [ -e "$file" ] checks if file exists
    # This prevents errors when no files match the pattern
    
    if [ -e "$file" ]; then
        
        # LESSON 6: STRING MANIPULATION
        # ------------------------------
        
        # Get file extension (everything after the last dot)
        # ${variable##pattern} removes the longest match from the beginning
        extension="${file##*.}"
        
        # Get directory path (everything before the last slash)
        # ${variable%pattern} removes the shortest match from the end
        directory="${file%/*}"
        
        # Get just the filename without path
        # basename extracts the filename from a full path
        old_filename=$(basename "$file")
        
        # LESSON 7: STRING INTERPOLATION
        # -------------------------------
        # ${variable} embeds the value of a variable in a string
        # You can also use $variable but ${variable} is safer
        
        new_name="${directory}/${PREFIX}_${counter}.${extension}"
        new_filename="${PREFIX}_${counter}.${extension}"
        
        # LESSON 8: COMMANDS
        # ------------------
        # mv = move/rename files
        # Syntax: mv <source> <destination>
        
        mv "$file" "$new_name"
        
        # Print feedback (with color!)
        echo "✓ Renamed: $old_filename → $new_filename"
        
        # LESSON 9: ARITHMETIC OPERATIONS
        # --------------------------------
        # ((expression)) performs arithmetic
        # ++ is increment operator (adds 1)
        
        ((counter++))
        # Alternative: counter=$((counter + 1))
    fi
done

echo ""
echo "✅ Successfully renamed $((counter - 1)) files!"

# ==============================================================================
# KEY BASH CONCEPTS COVERED:
# ==============================================================================
# 1. Shebang (#!/bin/bash) - tells system to use bash interpreter
# 2. Comments (#) - document your code
# 3. Variables ($VAR or ${VAR}) - store and retrieve values
# 4. Arguments ($1, $2, $#) - get user input
# 5. Conditionals (if/then/fi) - make decisions
# 6. File tests (-e, -d) - check file properties
# 7. Loops (for...do...done) - repeat actions
# 8. String manipulation (${var##pattern}, ${var%pattern}) - extract parts
# 9. Arithmetic ((expression)) - do math
# 10. Commands (mv, basename) - interact with system
# ==============================================================================
