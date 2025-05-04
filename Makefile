# --- Configuration ---
# Rust executable path
EXECUTABLE := ./target/release/Rasterizer_rust

# Input model
OBJ_FILE   := obj/simple/bunny.obj

# Output settings
MODEL_NAME := bunny_orbit_rust
OUTPUT_DIR := output_rust_$(MODEL_NAME)
WIDTH      := 1024
HEIGHT     := 1024

# Animation settings
FRAMES     := 120
FPS        := 30
VIDEO_NAME := $(MODEL_NAME).mp4

# Camera orbit settings
ORBIT_RADIUS := 2.5
ORBIT_HEIGHT := 0.5
LOOK_AT      := 0,0.1,0 # Point the camera looks at (Removed inner quotes)
CAMERA_UP    := 0,1,0   # Camera up direction (Removed inner quotes)

# Lighting settings (copied from the single command example)
LIGHT_TYPE   := directional
LIGHT_DIR    := "0,-1,0"
DIFFUSE      := 0.8
AMBIENT      := 0.15

# Other rendering flags (copied from the single command example)
NO_DEPTH     := --no-depth
NO_TEXTURE   := --no-texture
COLORIZE     := --colorize

# --- Derived Variables ---
# Generate frame numbers from 0 to FRAMES-1
FRAME_NUMS := $(shell seq 0 $(shell expr $(FRAMES) - 1))
# Define output PNG file paths with 3-digit padding for frame numbers
FRAME_FILES := $(patsubst %,$(OUTPUT_DIR)/frame_%03d_color.png,$(FRAME_NUMS))

# --- Targets ---
.PHONY: all frames video clean build

# Default target: build the executable and create the video
all: build video

# Target to build the Rust executable
build: $(EXECUTABLE)

$(EXECUTABLE): src/*.rs Cargo.toml Cargo.lock
	@echo "Building Rust executable..."
	cargo build --release

# Rule to create the output directory if it doesn't exist
# This is an order-only prerequisite for the frame rendering rule
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Rule to render a single frame
# Depends on the executable being built and the output directory existing
$(OUTPUT_DIR)/frame_%_color.png: $(EXECUTABLE) | $(OUTPUT_DIR)
	@frame_num=$*;\
	echo "Rendering frame $$frame_num of $(FRAMES)...";\
	# Use awk to calculate camera position for the orbit and generate arguments
	# Pass LOOK_AT without inner quotes, removed gsub in awk script
	camera_args=$$(awk -v frame=$$frame_num -v total=$(FRAMES) \
	                 -v radius=$(ORBIT_RADIUS) -v height=$(ORBIT_HEIGHT) \
	                 -v look_at=$(LOOK_AT) 'BEGIN { \
	                    pi = 3.141592653589793; \
	                    # Split look_at string directly (no need for gsub)
	                    split(look_at, la, ","); \
	                    # Calculate angle in radians for the current frame
	                    angle = frame * 2 * pi / total; \
	                    # Calculate camera coordinates based on orbit parameters and look_at point
	                    cam_x = la[1] + radius * cos(angle); \
	                    cam_y = la[2] + height; \
	                    cam_z = la[3] + radius * sin(angle); \
	                    # Output the camera position and output filename arguments for the executable
	                    # Frame number is padded to 3 digits (%03d)
	                    printf "--camera-from=\"%.6f,%.6f,%.6f\" --output=\"frame_%03d\"", cam_x, cam_y, cam_z, frame; \
	                 }'); \
	# Execute the renderer with all necessary arguments
	# Ensure LOOK_AT and CAMERA_UP passed to executable are quoted
	$(EXECUTABLE) \
	    --obj=$(OBJ_FILE) \
	    --output-dir=$(OUTPUT_DIR) \
	    --width=$(WIDTH) \
	    --height=$(HEIGHT) \
	    --camera-at="$(LOOK_AT)" \
	    --camera-up="$(CAMERA_UP)" \
	    --light-type=$(LIGHT_TYPE) \
	    --light-dir="$(LIGHT_DIR)" \
	    --diffuse=$(DIFFUSE) \
	    --ambient=$(AMBIENT) \
	    $(NO_DEPTH) \
	    $(NO_TEXTURE) \
	    $(COLORIZE) \
	    $$camera_args # Append the calculated camera arguments

# Target to render all frames (leverages Make's implicit parallelism if -j flag is used)
frames: $(FRAME_FILES)

# Rule to create the video from the rendered frames using ffmpeg
video: frames
	@echo "Creating video $(VIDEO_NAME)..."
	ffmpeg -y -framerate $(FPS) -i "$(OUTPUT_DIR)/frame_%03d_color.png" \
	       -c:v libx264 -pix_fmt yuv420p -vf "scale=$(WIDTH):-2" $(VIDEO_NAME)
	@echo "Video saved as $(VIDEO_NAME)"

# Rule to clean up generated files
clean:
	@echo "Cleaning output directory ($(OUTPUT_DIR)) and video ($(VIDEO_NAME))..."
	rm -rf $(OUTPUT_DIR)
	rm -f $(VIDEO_NAME)
	@echo "Clean complete."

# Remove the old single command execution block if it exists
# The following command block is now handled by the 'frames' and 'video' targets
# ./target/release/Rasterizer_rust \
#     --obj=obj/simple/bunny.obj \
#     --output-dir=output_rust_bunny_orbit_rust \
#     --width=1024 \
#     --height=1024 \
#     --camera-at="0,0.1,0" \
#     --camera-up="0,1,0" \
#     --light-type=directional \
#     --light-dir="0,-1,0" \
#     --diffuse=0.8 \
#     --ambient=0.15 \
#     --no-depth \
#     --no-texture \
#     --colorize \
#     --output="frame_061" \
#     --camera-from="-0.130840,0.5,-2.496574"
