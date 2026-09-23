#!/bin/bash

# Check if a path argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 path_to_images num_frames fps [num_workers]"
  exit 1
fi

# Set up working variables
DIR=$1
NUM_FRAMES=$2  # number of frames
FPS=$3         # frames per second
NUM_WORKERS=${4:-8}  # number of parallel jobs, default 8

# Prepare image folder
cd "$DIR" || { echo "Failed to cd into $DIR"; exit 1; }

# Function to process a file
task() {
   local file="$1"
   echo "Processing $file file..."

   filenameOnly=${file##*/}
   filenameNoExt=${filenameOnly%.*}
   echo "Filename without extension: $filenameNoExt"
   echo "Filename with extension: $filenameOnly"

   povray -H2160 -W3840 Quality=11 Antialias=on "$filenameOnly"
}
export -f task

# Generate the file list and process in parallel
seq -f "%02g" 0 "$NUM_FRAMES" | parallel -j "$NUM_WORKERS" task "moving_arm{}.pov"

# Create a video from the rendered images
echo "Creating video..."
ffmpeg -threads "$NUM_WORKERS" -r "$FPS" -i "./moving_arm%02d.png" -b:v 90M -vcodec mpeg4 "./output_video.mp4"

# Return to original folder
cd "$STARTFOLDER"
