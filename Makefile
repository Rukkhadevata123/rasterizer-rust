# --- 基础配置 ---
BUILD_TYPE   := release
CARGO_BUILD_CMD := cargo build --$(BUILD_TYPE)
EXECUTABLE := ./target/$(BUILD_TYPE)/rasterizer

# --- 图像与动画设置 ---
WIDTH		:= 2048
HEIGHT	   := 2048
TOTAL_FRAMES := 120
ANIM_FPS	 := 30

# --- 默认渲染参数 ---
CAMERA_UP	:= "0,1,0"
CAMERA_FOV   := 45.0
PROJECTION   := perspective
LIGHT_TYPE   := directional
AMBIENT	  := 0.2
AMBIENT_COLOR := "0.2,0.2,0.2"
DIFFUSE	  := 0.8
USE_PHONG	:= true
USE_PBR	  := false

# --- 模型配置 --- 
# 斯坦福兔子模型 (修正相机位置使camera_from和camera_at在同一水平面上)
BUNNY_MODEL  := obj/simple/bunny.obj
BUNNY_OUTPUT := output_bunny
BUNNY_CAMERA_FROM := "0,0.1,2.5"  # 修改Y坐标与目标点相同
BUNNY_CAMERA_AT := "0,0.1,0"
BUNNY_LIGHT_DIR := "0,-1,-1"

# 奶牛模型 (修正相机位置使camera_from和camera_at在同一水平面上)
SPOT_MODEL   := obj/models/spot/spot_triangulated.obj
SPOT_TEXTURE := obj/models/spot/spot_texture.png
SPOT_OUTPUT  := output_spot
SPOT_CAMERA_FROM := "0,0.5,3"  # 修改Y坐标与目标点相同
SPOT_CAMERA_AT := "0,0.5,0"
SPOT_LIGHT_DIR := "0.5,-1.0,-0.5"

# 岩石模型 (PBR演示，修正相机位置使camera_from和camera_at在同一水平面上)
ROCK_MODEL   := obj/models/rock/rock.obj
ROCK_TEXTURE := obj/models/rock/rock.png
ROCK_OUTPUT  := pbr_rock_render
ROCK_CAMERA_FROM := "3,0.5,3"  # 修改Y坐标与目标点相同
ROCK_CAMERA_AT := "0,0.5,0"
ROCK_LIGHT_POS := "5,5,2"

# --- 默认运行设置 ---
OBJ_FILE	:= $(BUNNY_MODEL)
OUTPUT_DIR  := $(BUNNY_OUTPUT)
CAMERA_FROM := $(BUNNY_CAMERA_FROM)
LOOK_AT	 := $(BUNNY_CAMERA_AT)
LIGHT_DIR   := $(BUNNY_LIGHT_DIR)
TEXTURE_FILE:= 

# --- 定义phony目标 ---
.PHONY: build run clean animate video test all pbr_rock bunny_orbit spot_orbit

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
	--ambient-color $(AMBIENT_COLOR) \
	--diffuse $(DIFFUSE) \
	$(if $(filter true,$(USE_PHONG)),--use-phong) \
	$(if $(filter true,$(USE_PBR)),--use-pbr) \
	$(if $(TEXTURE_FILE),--texture $(TEXTURE_FILE))

# --- 基础命令 ---
# 构建可执行文件
build:
	@echo "构建 $(BUILD_TYPE) 可执行文件..."
	$(CARGO_BUILD_CMD)

# 执行单帧渲染
run: build
	@echo "渲染单帧: $(OBJ_FILE)..."
	mkdir -p $(OUTPUT_DIR)
	$(EXECUTABLE) $(COMMON_ARGS) --output render
	@echo "渲染完成，输出到 $(OUTPUT_DIR)/render_*.png"

# 清理构建产物和输出目录
clean:
	@echo "清理项目..."
	cargo clean
	rm -rf output_* *.mp4
	@echo "清理完成"

# 运行测试
test:
	@echo "运行测试..."
	cargo test

# --- 动画命令 ---
# 渲染动画帧序列
animate: build
	@echo "渲染动画: $(OBJ_FILE)..."
	mkdir -p $(OUTPUT_DIR)
	$(EXECUTABLE) $(COMMON_ARGS) --animate --total-frames $(TOTAL_FRAMES)
	@echo "动画渲染完成，输出到 $(OUTPUT_DIR)/frame_*.png"

# 从动画帧创建视频
video: 
	@echo "生成视频: $(OUTPUT_DIR).mp4"
	ffmpeg -y -framerate $(ANIM_FPS) -i $(OUTPUT_DIR)/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p $(OUTPUT_DIR).mp4
	@echo "视频生成完成: $(OUTPUT_DIR).mp4"

# --- 兔子模型演示 ---
# 渲染兔子轨道动画
bunny_orbit: build
	@echo "渲染兔子轨道动画..."
	mkdir -p output_bunny_orbit
	$(EXECUTABLE) --obj $(BUNNY_MODEL) \
		--output-dir output_bunny_orbit \
		--width $(WIDTH) --height $(HEIGHT) \
		--camera-from $(BUNNY_CAMERA_FROM) --camera-at $(BUNNY_CAMERA_AT) \
		--camera-up $(CAMERA_UP) --camera-fov $(CAMERA_FOV) \
		--light-type $(LIGHT_TYPE) --light-dir $(BUNNY_LIGHT_DIR) \
		--ambient $(AMBIENT) --diffuse $(DIFFUSE) \
		--use-phong \
		--animate --total-frames $(TOTAL_FRAMES)
	@echo "渲染完成，输出到 output_bunny_orbit/"
	@echo "生成视频，请运行: make bunny_orbit_video"

# 生成兔子轨道视频
bunny_orbit_video:
	@echo "生成兔子轨道视频..."
	ffmpeg -y -framerate $(ANIM_FPS) -i output_bunny_orbit/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p demo/bunny_orbit.mp4
	@echo "视频生成完成: demo/bunny_orbit.mp4"

# --- Spot模型演示 ---
# 渲染Spot轨道动画
spot_orbit: build
	@echo "渲染Spot轨道动画..."
	mkdir -p output_spot_orbit
	$(EXECUTABLE) --obj $(SPOT_MODEL) \
		--texture $(SPOT_TEXTURE) \
		--output-dir output_spot_orbit \
		--width $(WIDTH) --height $(HEIGHT) \
		--camera-from $(SPOT_CAMERA_FROM) --camera-at $(SPOT_CAMERA_AT) \
		--use-phong \
		--animate --total-frames $(TOTAL_FRAMES)
	@echo "渲染完成，输出到 output_spot_orbit/"
	@echo "生成视频，请运行: make spot_orbit_video"

# 生成Spot轨道视频
spot_orbit_video:
	@echo "生成Spot轨道视频..."
	mkdir -p demo
	ffmpeg -y -framerate $(ANIM_FPS) -i output_spot_orbit/frame_%03d_color.png -c:v libx264 -pix_fmt yuv420p demo/spot_orbit.mp4
	@echo "视频生成完成: demo/spot_orbit.mp4"

# --- PBR 高级渲染演示 ---
# 岩石模型PBR渲染
pbr_rock: build
	@echo "渲染岩石模型PBR演示..."
	mkdir -p $(ROCK_OUTPUT)
	$(EXECUTABLE) --obj $(ROCK_MODEL) \
		--texture $(ROCK_TEXTURE) \
		--output-dir $(ROCK_OUTPUT) \
		--width $(WIDTH) --height $(HEIGHT) \
		--camera-from $(ROCK_CAMERA_FROM) --camera-at $(ROCK_CAMERA_AT) \
		--camera-up $(CAMERA_UP) --camera-fov $(CAMERA_FOV) \
		--light-type point --light-pos "3,5,2" \
		--light-atten "1.0,0.03,0.01" \
		--ambient 0.6 --ambient-color "0.5,0.5,0.55" \
		--diffuse 1.5 \
		--use-phong --use-pbr \
		--metallic 0.6 --roughness 0.25 \
		--base-color "0.95,0.95,0.8" \
		--ambient-occlusion 0.9 \
		--emissive "0.25,0.15,0.08" \
		--output pbr_rock
	@echo "PBR渲染完成，输出到 $(ROCK_OUTPUT)/pbr_rock_*.png"

# --- 综合命令 ---
# 运行所有演示目标
all: build bunny_orbit spot_orbit pbr_rock
	@echo "所有演示渲染完成"