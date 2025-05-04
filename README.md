# Rasterizer_rust

一个使用Rust语言实现的高性能软件光栅化器，支持3D模型渲染、纹理映射和光照效果。

## 项目概述

本项目是一个从零开始实现的软件光栅化渲染器，无需依赖OpenGL等图形API，完全由CPU计算所有渲染步骤。渲染器利用Rust的安全性和并行计算能力，实现了高效的三角形光栅化、深度测试、光照计算和纹理映射等功能。

## 主要功能

- **3D模型加载**: 支持OBJ格式模型的加载和渲染
- **多种投影方式**: 支持透视投影和正交投影
- **光照模型**: 实现了环境光、方向光和点光源，以及Blinn-Phong着色模型
- **纹理映射**: 支持图片纹理加载和UV映射
- **多种着色方式**: 支持平面着色(Flat Shading)和Phong着色(逐像素光照)
- **Z缓冲**: 实现了线程安全的深度测试
- **动画渲染**: 支持相机轨道动画生成
- **深度可视化**: 可输出深度图

## 渲染管线

渲染器实现了完整的图形渲染管线，包括：

1. **模型加载**: 从OBJ文件加载顶点、法线、纹理坐标和材质信息
2. **模型标准化**: 将模型居中并缩放到标准尺寸
3. **顶点变换**:
   - 世界变换: 模型空间 → 世界空间
   - 视图变换: 世界空间 → 相机空间
   - 投影变换: 相机空间 → 裁剪空间
   - 透视除法: 裁剪空间 → NDC(标准化设备坐标)
   - 视口变换: NDC → 屏幕坐标
4. **三角形设置**: 准备三角形数据，包括背面剔除
5. **三角形光栅化**:
   - 计算包围盒
   - 逐像素处理
   - 重心坐标计算
   - 深度插值与测试
   - 属性插值(纹理坐标、法线等)
6. **片段着色**:
   - 纹理采样
   - 光照计算(环境光、漫反射、镜面反射)
7. **帧缓冲更新**: 将计算结果写入颜色缓冲区

## 关键技术实现

### 1. 重心坐标插值

使用重心坐标进行顶点属性的插值，并实现了透视校正插值，确保纹理映射和其他属性在透视投影下的正确性：

```rust
pub fn interpolate_texcoords(
    bary: Vector3<f32>,
    tc1: Vector2<f32>,
    tc2: Vector2<f32>,
    tc3: Vector2<f32>,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> Vector2<f32> {
    // 透视校正插值实现...
}
```

### 2. 光照计算

实现了Blinn-Phong光照模型，支持环境光、漫反射和镜面反射计算：

```rust
pub fn calculate_blinn_phong(
    point: Point3<f32>,
    normal: Vector3<f32>,
    view_dir: Vector3<f32>,
    light: &Light,
    material: &SimpleMaterial,
) -> Color {
    // Blinn-Phong 光照计算...
}
```

### 3. 线程安全的Z缓冲

使用原子操作确保多线程渲染时深度缓冲的一致性：

```rust
let current_depth_atomic = &depth_buffer[pixel_index];
let old_depth_before_update = current_depth_atomic
    .fetch_min(interpolated_depth, Ordering::Relaxed);
```

### 4. 并行三角形处理

使用Rayon库实现三角形并行光栅化，大幅提升渲染性能：

```rust
triangles_to_render.par_iter().for_each(|triangle_data| {
    rasterize_triangle(
        triangle_data,
        self.frame_buffer.width,
        self.frame_buffer.height,
        &self.frame_buffer.depth_buffer,
        &self.frame_buffer.color_buffer,
    );
});
```

## 使用方法

### 编译

```bash
# 使用Cargo构建
cargo build --release

# 或使用Makefile构建
make build
```

### 使用Cargo运行

基本渲染：

```bash
cargo run --release -- --obj obj/bunny/bunny2k_f.obj --width 800 --height 600 --output bunny
```

带纹理渲染：

```bash
cargo run --release -- --obj obj/models/your_model.obj --width 1024 --height 768 --output textured_model --use-phong
```

生成轨道动画：

```bash
cargo run --release -- --obj obj/bunny/bunny2k_f.obj --width 800 --height 600 --output bunny_orbit --animate
```

### 使用Makefile运行

项目包含了预配置的Makefile，提供了多种便捷的命令：

```bash
# 渲染单帧图像（使用默认配置）
make run

# 渲染动画序列
make animate

# 运行斯坦福兔子模型演示（带轨道动画）
make bunny_demo

# 为兔子动画生成视频（需要安装ffmpeg）
make bunny_video

# 运行带纹理的奶牛模型演示
make spot_demo

# 为奶牛模型动画生成视频
make spot_video

# 清理构建和输出文件
make clean
```

Makefile提供了多种可配置选项，如渲染分辨率、相机参数和光照设置等。可以直接编辑Makefile或通过命令行覆盖这些设置：

```bash
# 自定义分辨率和输出目录
make run WIDTH=1920 HEIGHT=1080 OUTPUT_DIR=my_renders

# 使用特定模型和纹理
make run OBJ_FILE=obj/models/your_model.obj TEXTURE_FILE=obj/models/your_texture.png
```

## 项目结构

- `src/main.rs`: 程序入口和命令行参数处理
- `src/renderer.rs`: 渲染器核心，负责整体渲染流程
- `src/rasterizer.rs`: 三角形光栅化实现
- `src/interpolation.rs`: 属性插值功能
- `src/lighting.rs`: 光照模型实现
- `src/camera.rs`: 相机设置和矩阵计算
- `src/texture_utils.rs`: 纹理加载和采样
- `src/loaders.rs`: OBJ模型和材质加载
- `src/transform.rs`: 坐标变换函数
- `src/color_utils.rs`: 颜色处理工具
- `src/model_types.rs`: 模型数据结构定义

## 未来改进方向

- 优化深度缓冲区的并行访问性能
- 实现更多高级着色模型如PBR渲染
- 添加更多的图元支持(点、线)
- 实现层次包围盒加速三角形遍历
- 添加后处理效果如抗锯齿、景深等
