# --- Configuration ---
# 构建类型: debug 或 release
BUILD_TYPE   := release
CARGO_BUILD_CMD := cargo build --$(BUILD_TYPE)
EXECUTABLE := ./target/$(BUILD_TYPE)/Rasterizer_rust

# --- 模型配置 --- 
# 斯坦福兔子模型
BUNNY_MODEL := obj/simple/bunny.obj
BUNNY_OUTPUT_DIR := output_bunny_orbit
BUNNY_VIDEO_NAME := bunny_orbit.mp4
BUNNY_CAMERA_FROM := "0,0.5,2.5"
BUNNY_CAMERA_AT := "0,0.1,0"
BUNNY_LIGHT_DIR := "0,-1,-1"

# 奶牛模型 (有纹理)
SPOT_MODEL := obj/models/spot/spot_triangulated.obj
SPOT_TEXTURE := obj/models/spot/spot_texture.png
SPOT_OUTPUT_DIR := output_spot_orbit
SPOT_VIDEO_NAME := spot_orbit.mp4
SPOT_CAMERA_FROM := "0,1.5,3"
SPOT_CAMERA_AT := "0,0.5,0"
SPOT_LIGHT_DIR := "0.5,-1.0,-0.5"

# --- 通用渲染设置 ---
# 图像尺寸
WIDTH      := 2048
HEIGHT     := 2048
# 通用相机参数
CAMERA_UP    := "0,1,0"
CAMERA_FOV   := 45.0
PROJECTION   := perspective
# 光照参数
LIGHT_TYPE   := directional
AMBIENT      := 0.2
DIFFUSE      := 0.8
# 渲染选项
NO_ZBUFFER   := false
COLORIZE     := false
NO_DEPTH     := false
NO_TEXTURE   := false
NO_LIGHTING  := false
USE_PHONG    := true     # 是否使用 Phong 着色（逐像素光照）

# --- 动画设置 ---
ANIM_FPS    := 30

# --- 默认运行模型 ---
OBJ_FILE    := $(BUNNY_MODEL)
OUTPUT_DIR  := $(BUNNY_OUTPUT_DIR)
OUTPUT_NAME := bunny_render
CAMERA_FROM := $(BUNNY_CAMERA_FROM)
LOOK_AT     := $(BUNNY_CAMERA_AT)
LIGHT_DIR   := $(BUNNY_LIGHT_DIR)
TEXTURE_FILE:= 

# --- 编译与渲染目标 ---
.PHONY: build run clean animate video test bunny_demo spot_demo

# 构建可执行文件
build:
	@echo "Building $(BUILD_TYPE) executable..."
	$(CARGO_BUILD_CMD)

# 通用渲染参数
COMMON_ARGS = \
	-o $(OBJ_FILE) \
	--output-dir $(OUTPUT_DIR) \
	--width $(WIDTH) \
	--height $(HEIGHT) \
	--projection $(PROJECTION) \
	--camera-from $(CAMERA_FROM) \
	--camera-at $(LOOK_AT) \
	--camera-up $(CAMERA_UP) \
	--camera-fov $(CAMERA_FOV) \
	--light-type $(LIGHT_TYPE) \
	--light-dir $(LIGHT_DIR) \
	--ambient $(AMBIENT) \
	--diffuse $(DIFFUSE) \
	$(if $(filter true,$(NO_ZBUFFER)),--no-zbuffer) \
	$(if $(filter true,$(COLORIZE)),--colorize) \
	$(if $(filter true,$(NO_DEPTH)),--no-depth) \
	$(if $(filter true,$(NO_LIGHTING)),--no-lighting) \
	$(if $(filter true,$(NO_TEXTURE)),--no-texture) \
	$(if $(filter true,$(USE_PHONG)),--use-phong) \
	$(if $(TEXTURE_FILE),--texture $(TEXTURE_FILE))

# 执行单帧渲染
run: build
	@echo "Running single frame render for $(OBJ_FILE)..."
	$(EXECUTABLE) $(COMMON_ARGS) --output $(OUTPUT_NAME)
	@echo "Single frame render complete. Output in $(OUTPUT_DIR)/$(OUTPUT_NAME)_*.png"

# 渲染动画帧序列
animate: build
	@echo "Running animation for $(OBJ_FILE)..."
	$(EXECUTABLE) $(COMMON_ARGS) --animate
	@echo "Animation rendering complete. Output frames in $(OUTPUT_DIR)/frame_*.png"

# 从动画帧创建视频 (需要 ffmpeg)
video:
	@echo "Creating video from frames in $(OUTPUT_DIR)..."
	ffmpeg -y -framerate $(ANIM_FPS) -i $(OUTPUT_DIR)/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p $(VIDEO_NAME)
	@echo "Video creation complete: $(VIDEO_NAME)"

# 兔子模型演示
bunny_demo: build
	@echo "Running Stanford Bunny animation demo..."
	$(EXECUTABLE) -o $(BUNNY_MODEL) \
		--output-dir $(BUNNY_OUTPUT_DIR) \
		--width $(WIDTH) --height $(HEIGHT) \
		--camera-from $(BUNNY_CAMERA_FROM) --camera-at $(BUNNY_CAMERA_AT) \
		--camera-up $(CAMERA_UP) --camera-fov $(CAMERA_FOV) \
		--light-type $(LIGHT_TYPE) --light-dir $(BUNNY_LIGHT_DIR) \
		--ambient $(AMBIENT) --diffuse $(DIFFUSE) \
		--animate
	@echo "Bunny animation complete. Output frames in $(BUNNY_OUTPUT_DIR)/"
	@echo "To create video: make bunny_video"

# 兔子模型视频生成
bunny_video:
	@echo "Creating Stanford Bunny video..."
	ffmpeg -y -framerate $(ANIM_FPS) -i $(BUNNY_OUTPUT_DIR)/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p $(BUNNY_VIDEO_NAME)
	@echo "Video creation complete: $(BUNNY_VIDEO_NAME)"

# 奶牛模型演示
spot_demo: build
	@echo "Running Spot (cow) model animation demo with texture..."
	$(EXECUTABLE) -o $(SPOT_MODEL) \
		--texture $(SPOT_TEXTURE) \
		--output-dir $(SPOT_OUTPUT_DIR) \
		--width $(WIDTH) --height $(HEIGHT) \
		--camera-from $(SPOT_CAMERA_FROM) --camera-at $(SPOT_CAMERA_AT) \
		--camera-up $(CAMERA_UP) --camera-fov $(CAMERA_FOV) \
		--light-type $(LIGHT_TYPE) --light-dir $(SPOT_LIGHT_DIR) \
		--ambient $(AMBIENT) --diffuse $(DIFFUSE) \
		--animate
	@echo "Spot animation complete. Output frames in $(SPOT_OUTPUT_DIR)/"
	@echo "To create video: make spot_video"

# 奶牛模型视频生成
spot_video:
	@echo "Creating Spot (cow) video..."
	ffmpeg -y -framerate $(ANIM_FPS) -i $(SPOT_OUTPUT_DIR)/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p $(SPOT_VIDEO_NAME)
	@echo "Video creation complete: $(SPOT_VIDEO_NAME)"

# 清理构建产物和输出目录/视频
clean:
	@echo "Cleaning build artifacts and output..."
	cargo clean
	rm -rf $(BUNNY_OUTPUT_DIR) $(SPOT_OUTPUT_DIR)
	rm -f $(BUNNY_VIDEO_NAME) $(SPOT_VIDEO_NAME)
	@echo "Clean complete."

# 运行测试
test:
	@echo "Running tests..."
	cargo test
