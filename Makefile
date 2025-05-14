# --- 基础配置 ---
# 构建类型: debug 或 release
BUILD_TYPE   := release
CARGO_BUILD_CMD := cargo build --$(BUILD_TYPE)
EXECUTABLE := ./target/$(BUILD_TYPE)/Rasterizer_rust

# --- 图像与动画设置 ---
# 图像尺寸
WIDTH      := 2048
HEIGHT     := 2048
# 设置动画帧数和帧率
TOTAL_FRAMES := 120
ANIM_FPS    := 30

# --- 相机基础设置 ---
# 通用相机参数
CAMERA_UP    := "0,1,0"
CAMERA_FOV   := 45.0
PROJECTION   := perspective

# --- 渲染选项设置 ---
# 光照参数
LIGHT_TYPE   := directional
AMBIENT      := 0.2
DIFFUSE      := 0.8
# 渲染功能开关
NO_ZBUFFER   := false
COLORIZE     := false
NO_DEPTH     := false
NO_TEXTURE   := false
NO_LIGHTING  := false
USE_PHONG    := true     # 是否使用 Phong 着色（逐像素光照）
USE_PBR      := false    # 是否使用基于物理的渲染(PBR)
NO_GAMMA     := false    # 是否禁用gamma矫正

# --- PBR材质参数 ---
METALLIC    := 0.8     # 金属度 (0.0-1.0)
ROUGHNESS   := 0.2     # 粗糙度 (0.0-1.0)
BASE_COLOR  := "0.8,0.8,0.8"  # 基础颜色 (r,g,b)
AMBIENT_OCCLUSION := 1.0      # 环境光遮蔽 (0.0-1.0)
EMISSIVE    := "0.0,0.0,0.0"  # 自发光颜色 (r,g,b)

# --- 模型配置 --- 
# 1. 斯坦福兔子模型
BUNNY_MODEL := obj/simple/bunny.obj
BUNNY_OUTPUT_DIR := output_bunny_orbit
BUNNY_VIDEO_NAME := bunny_orbit.mp4
BUNNY_CAMERA_FROM := "0,0.5,2.5"
BUNNY_CAMERA_AT := "0,0.1,0"
BUNNY_LIGHT_DIR := "0,-1,-1"

# 2. 奶牛模型 (有纹理)
SPOT_MODEL := obj/models/spot/spot_triangulated.obj
SPOT_TEXTURE := obj/models/spot/spot_texture.png
SPOT_OUTPUT_DIR := output_spot_orbit
SPOT_VIDEO_NAME := spot_orbit.mp4
SPOT_CAMERA_FROM := "0,1.5,3"
SPOT_CAMERA_AT := "0,0.5,0"
SPOT_LIGHT_DIR := "0.5,-1.0,-0.5"

# 3. 球体模型 (用于PBR和金属度测试)
SPHERE_MODEL := obj/collect1/sphere.obj
SPHERE_OUTPUT_DIR := output_sphere
SPHERE_CAMERA_FROM := "2,2,2"
SPHERE_CAMERA_AT := "0,0,0"
SPHERE_LIGHT_POS := "5,5,5"

# --- 默认运行设置 ---
OBJ_FILE    := $(BUNNY_MODEL)
OUTPUT_DIR  := $(BUNNY_OUTPUT_DIR)
OUTPUT_NAME := bunny_render
CAMERA_FROM := $(BUNNY_CAMERA_FROM)
LOOK_AT     := $(BUNNY_CAMERA_AT)
LIGHT_DIR   := $(BUNNY_LIGHT_DIR)
TEXTURE_FILE:= 

# 自动检测模型类型并设置相应目录
ifeq ($(findstring spot,$(OBJ_FILE)),spot)
    AUTO_OUTPUT_DIR := $(SPOT_OUTPUT_DIR)
    AUTO_CAMERA_FROM := $(SPOT_CAMERA_FROM)
    AUTO_LOOK_AT := $(SPOT_CAMERA_AT)
    AUTO_LIGHT_DIR := $(SPOT_LIGHT_DIR)
else
    AUTO_OUTPUT_DIR := $(BUNNY_OUTPUT_DIR)
    AUTO_CAMERA_FROM := $(BUNNY_CAMERA_FROM)
    AUTO_LOOK_AT := $(BUNNY_CAMERA_AT)
    AUTO_LIGHT_DIR := $(BUNNY_LIGHT_DIR)
endif

# --- 定义phony目标 ---
.PHONY: build run clean animate video test \
        bunny_demo spot_demo bunny_gamma bunny_ortho \
        multi_objects spot_phong pbr_demo benchmark \
        profile sphere_metal all_demos

# --- 通用渲染参数 ---
COMMON_ARGS = \
	--obj $(OBJ_FILE) \
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
	$(if $(filter true,$(USE_PBR)),--use-pbr) \
	$(if $(filter true,$(NO_GAMMA)),--no-gamma) \
	$(if $(METALLIC),--metallic $(METALLIC)) \
	$(if $(ROUGHNESS),--roughness $(ROUGHNESS)) \
	$(if $(BASE_COLOR),--base-color $(BASE_COLOR)) \
	$(if $(AMBIENT_OCCLUSION),--ambient-occlusion $(AMBIENT_OCCLUSION)) \
	$(if $(EMISSIVE),--emissive $(EMISSIVE)) \
	$(if $(TEXTURE_FILE),--texture $(TEXTURE_FILE)) \
	$(if $(OBJECT_COUNT),--object-count $(OBJECT_COUNT)) \
	$(if $(SHOW_DEBUG),--show-debug-info)

# --- 基础构建与渲染目标 ---
# 构建可执行文件
build:
	@echo "Building $(BUILD_TYPE) executable..."
	$(CARGO_BUILD_CMD)

# 执行单帧渲染
run: build
	@echo "Running single frame render for $(OBJ_FILE)..."
	@# 根据OBJ_FILE自动设置输出目录，除非明确指定了OUTPUT_DIR
	@if [ "$(origin OUTPUT_DIR)" = "default" ]; then \
		$(eval OUTPUT_DIR=$(AUTO_OUTPUT_DIR)) \
		$(eval CAMERA_FROM=$(AUTO_CAMERA_FROM)) \
		$(eval LOOK_AT=$(AUTO_LOOK_AT)) \
		$(eval LIGHT_DIR=$(AUTO_LIGHT_DIR)) \
		echo "自动选择输出目录: $(OUTPUT_DIR)"; \
	fi
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

# --- 示例渲染目标 ---
# 1. 兔子模型演示与视频生成
bunny_demo: build
	@echo "Running Stanford Bunny animation demo..."
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir $(BUNNY_OUTPUT_DIR) \
		--width $(WIDTH) --height $(HEIGHT) \
		--camera-from $(BUNNY_CAMERA_FROM) --camera-at $(BUNNY_CAMERA_AT) \
		--camera-up $(CAMERA_UP) --camera-fov $(CAMERA_FOV) \
		--light-type $(LIGHT_TYPE) --light-dir $(BUNNY_LIGHT_DIR) \
		--ambient $(AMBIENT) --diffuse $(DIFFUSE) \
		--animate
	@echo "Bunny animation complete. Output frames in $(BUNNY_OUTPUT_DIR)/"
	@echo "To create video: make bunny_video"

bunny_video:
	@echo "Creating Stanford Bunny video..."
	ffmpeg -y -framerate $(ANIM_FPS) -i $(BUNNY_OUTPUT_DIR)/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p $(BUNNY_VIDEO_NAME)
	@echo "Video creation complete: $(BUNNY_VIDEO_NAME)"

# 2. 奶牛模型演示与视频生成
spot_demo: build
	@echo "Running Spot (cow) model animation demo with texture..."
	$(EXECUTABLE) --obj $(SPOT_MODEL) \
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

spot_video:
	@echo "Creating Spot (cow) video..."
	ffmpeg -y -framerate $(ANIM_FPS) -i $(SPOT_OUTPUT_DIR)/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p $(SPOT_VIDEO_NAME)
	@echo "Video creation complete: $(SPOT_VIDEO_NAME)"

# --- 对比演示目标 ---
# 1. Gamma校正对比
bunny_gamma: build
	@echo "Running Stanford Bunny with gamma correction..."
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir gamma_comparison \
		--width 1024 --height 1024 \
		--camera-from $(BUNNY_CAMERA_FROM) --camera-at $(BUNNY_CAMERA_AT) \
		--output bunny_with_gamma
	@echo "Then running the same model without gamma correction..."
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir gamma_comparison \
		--width 1024 --height 1024 \
		--camera-from $(BUNNY_CAMERA_FROM) --camera-at $(BUNNY_CAMERA_AT) \
		--no-gamma \
		--output bunny_no_gamma
	@echo "Gamma comparison complete. Check gamma_comparison directory."

# 2. 投影类型对比
bunny_ortho: build
	@echo "Running Stanford Bunny with orthographic projection..."
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir projection_comparison \
		--width 1024 --height 1024 \
		--camera-from "0,0.5,5" --camera-at $(BUNNY_CAMERA_AT) \
		--projection orthographic \
		--output bunny_ortho
	@echo "Then running the same model with perspective projection..."
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir projection_comparison \
		--width 1024 --height 1024 \
		--camera-from "0,0.5,5" --camera-at $(BUNNY_CAMERA_AT) \
		--projection perspective \
		--output bunny_perspective
	@echo "Projection comparison complete. Check projection_comparison directory."

# 3. 着色模型对比
spot_phong: build
	@echo "Running Spot model with Phong shading..."
	$(EXECUTABLE) --obj $(SPOT_MODEL) \
		--texture $(SPOT_TEXTURE) \
		--output-dir shading_comparison \
		--width 1024 --height 1024 \
		--camera-from $(SPOT_CAMERA_FROM) --camera-at $(SPOT_CAMERA_AT) \
		--use-phong \
		--output spot_phong
	@echo "Then running the same model without Phong shading (flat shading)..."
	$(EXECUTABLE) --obj $(SPOT_MODEL) \
		--texture $(SPOT_TEXTURE) \
		--output-dir shading_comparison \
		--width 1024 --height 1024 \
		--camera-from $(SPOT_CAMERA_FROM) --camera-at $(SPOT_CAMERA_AT) \
		--output spot_flat
	@echo "Shading comparison complete. Check shading_comparison directory."

# 4. 渲染模型对比
pbr_demo: build
	@echo "Running PBR rendering demo..."
	$(EXECUTABLE) --obj $(SPOT_MODEL) \
		--texture $(SPOT_TEXTURE) \
		--output-dir pbr_comparison \
		--width 1024 --height 1024 \
		--camera-from $(SPOT_CAMERA_FROM) --camera-at $(SPOT_CAMERA_AT) \
		--use-pbr \
		--output spot_pbr
	@echo "Then running the same model with traditional Blinn-Phong shading..."
	$(EXECUTABLE) --obj $(SPOT_MODEL) \
		--texture $(SPOT_TEXTURE) \
		--output-dir pbr_comparison \
		--width 1024 --height 1024 \
		--camera-from $(SPOT_CAMERA_FROM) --camera-at $(SPOT_CAMERA_AT) \
		--use-phong \
		--output spot_blinn_phong
	@echo "PBR comparison complete. Check pbr_comparison directory."

# 5. 材质金属性对比
sphere_metal: build
	@echo "Running sphere model with metallic material..."
	$(EXECUTABLE) --obj $(SPHERE_MODEL) \
		--output-dir $(SPHERE_OUTPUT_DIR) \
		--width 1024 --height 1024 \
		--camera-from $(SPHERE_CAMERA_FROM) --camera-at $(SPHERE_CAMERA_AT) \
		--camera-up $(CAMERA_UP) --camera-fov $(CAMERA_FOV) \
		--light-type point --light-pos $(SPHERE_LIGHT_POS) \
		--ambient $(AMBIENT) --diffuse $(DIFFUSE) \
		--use-pbr --metallic $(METALLIC) --roughness $(ROUGHNESS) \
		--use-phong \
		--output sphere_metal
	@echo "Then running the same model with lower metallic value..."
	$(EXECUTABLE) --obj $(SPHERE_MODEL) \
		--output-dir $(SPHERE_OUTPUT_DIR) \
		--width 1024 --height 1024 \
		--camera-from $(SPHERE_CAMERA_FROM) --camera-at $(SPHERE_CAMERA_AT) \
		--camera-up $(CAMERA_UP) --camera-fov $(CAMERA_FOV) \
		--light-type point --light-pos $(SPHERE_LIGHT_POS) \
		--ambient $(AMBIENT) --diffuse $(DIFFUSE) \
		--use-pbr --metallic 0.2 --roughness $(ROUGHNESS) \
		--use-phong \
		--output sphere_nonmetal
	@echo "Metallic comparison complete. Check $(SPHERE_OUTPUT_DIR) directory."

# --- 高级示例 ---
# 1. 多对象场景
multi_objects: build
	@echo "Running demo with multiple instances of Stanford Bunny..."
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir multi_object_demo \
		--width 1024 --height 1024 \
		--camera-from "0,1.5,4" --camera-at "0,0,0" \
		--object-count 5 \
		--animate
	@echo "Multi-object animation complete. Check multi_object_demo directory."

# 2. 旋转演示
rotation_demo: build
	@echo "Running rotation comparison demo..."
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir rotation_comparison \
		--width 1024 --height 1024 \
		--camera-from "0,1.5,4" --camera-at "0,0,0" \
		--object-count 4 \
		--show-debug-info \
		--animate
	@echo "Rotation comparison complete. Check rotation_comparison directory."

# --- 性能工具 ---
# 1. 性能基准测试
benchmark: build
	@echo "Running performance benchmark..."
	time $(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir benchmark \
		--width 2048 --height 2048 \
		--camera-from $(BUNNY_CAMERA_FROM) --camera-at $(BUNNY_CAMERA_AT) \
		--use-phong \
		--output benchmark_phong
	@echo "Then running without Phong shading..."
	time $(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir benchmark \
		--width 2048 --height 2048 \
		--camera-from $(BUNNY_CAMERA_FROM) --camera-at $(BUNNY_CAMERA_AT) \
		--output benchmark_flat
	@echo "Benchmark complete. Check benchmark directory."

# 2. 性能分析
profile: build
	@echo "Running with profiling enabled..."
	@# 要求安装cargo-flamegraph
	cargo flamegraph --bin Rasterizer_rust -- --obj $(BUNNY_MODEL) \
		--width 1024 --height 1024 \
		--output profile_result
	@echo "Profile complete. Check flamegraph.svg for results."

# --- 批量演示 ---
# 运行所有演示目标
all_demos: build bunny_demo spot_demo bunny_gamma bunny_ortho multi_objects spot_phong pbr_demo rotation_demo sphere_metal
	@echo "All demos completed. Check output directories for results."
