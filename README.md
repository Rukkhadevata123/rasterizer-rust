# Rust 高性能光栅化渲染器 v2.2 🎨

一个功能完备的软件光栅化渲染器，采用**TOML驱动配置**和**现代化GUI界面**。支持从基础几何渲染到**高级PBR材质系统**、**增强次表面散射**、多光源系统、实时相机交互、配置文件管理等专业级渲染功能。

[![Rust Version](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/Rukkhadevata123/Rasterizer_rust)

## 🔥 v2.2 核心特性

### 🎨 **增强的材质系统**

- **� 扩展PBR参数** - 次表面散射、各向异性、法线强度控制
- **✨ 次表面散射** - 皮肤、蜡烛、大理石等半透明材质效果
- **🌟 各向异性反射** - 金属拉丝、织物等方向性材质
- **📐 可调法线强度** - 精细控制表面细节和凹凸效果
- **💎 Phong增强** - 独立漫反射和镜面反射强度控制

### �📝 **TOML 配置驱动**

- **完整的TOML配置支持** - 所有渲染参数均可通过配置文件设置
- **配置文件管理** - 一键加载/保存配置，示例配置生成
- **参数验证系统** - 智能检测并提示配置错误
- **向后兼容** - CLI参数与TOML配置无缝集成

### 🖥️ **现代化GUI界面**

- **专业级相机交互** - 鼠标拖拽、Shift+轨道旋转、滚轮缩放
- **实时参数调整** - 所见即所得的参数编辑体验
- **多面板设计** - 文件管理、渲染设置、材质调整、动画控制
- **中文界面支持** - 完整的本地化用户界面

### ⚡ **高性能渲染引擎**

- **多线程光栅化** - 充分利用现代多核CPU性能
- **智能剔除系统** - 背面剔除、视锥剔除、小三角形剔除
- **增强AO算法** - 基于法线、边缘、曲率的高级环境光遮蔽
- **软阴影效果** - 多光源软阴影，可调强度控制

### 🎬 **动画与视频系统**

- **实时动画渲染** - 支持相机轨道和物体旋转动画
- **预渲染模式** - 预计算帧序列，确保流畅播放
- **视频生成** - 集成FFmpeg，一键生成MP4动画
- **帧率统计** - 实时FPS监控和性能分析

## 目录

- [安装与构建](#安装与构建)
- [快速开始](#快速开始)
- [配置文件详解](#配置文件详解)
- [材质系统详解](#材质系统详解)
- [GUI使用指南](#gui使用指南)
- [命令行模式](#命令行模式)
- [渲染管线](#渲染管线)
- [项目架构](#项目架构)
- [示例与教程](#示例与教程)

## 安装与构建

### 环境要求

- **Rust**: 1.81+ (推荐最新稳定版)
- **依赖库**: 自动通过Cargo管理
- **FFmpeg**: (可选) 用于视频生成功能

#### 安装FFmpeg

**Windows**:

```powershell
# 使用Chocolatey
choco install ffmpeg

# 使用Winget
winget install Gyan.FFmpeg

# 或从官网下载并添加到PATH
# https://ffmpeg.org/download.html
```

**macOS**:

```bash
brew install ffmpeg
```

**Ubuntu/Debian**:

```bash
sudo apt-get update
sudo apt-get install ffmpeg libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev pkg-config
```

### 构建步骤

```bash
# 1. 克隆项目
git clone https://github.com/Rukkhadevata123/Rasterizer_rust
cd Rasterizer_rust

# 2. 构建 (开发模式)
cargo build

# 3. 构建 (发布模式，推荐)
cargo build --release

# 4. 运行
cargo run --release
```

## 快速开始

### 🚀 GUI模式 (推荐)

```bash
# 启动GUI (默认配置)
cargo run --release

# 从配置文件启动GUI
cargo run --release -- --config my_config.toml

# 使用示例配置启动GUI
cargo run --release -- --use-example-config
```

### ⚡ 命令行模式

```bash
# 生成示例配置文件
cargo run --release -- --use-example-config --headless

# 使用配置文件进行无头渲染
cargo run --release -- --config example_config.toml --headless
```

## 配置文件详解

### 基础配置结构

```toml
# config.toml - 完整配置示例

[files]
obj = "obj/models/spot/spot_triangulated.obj"
output = "my_render"
output_dir = "output"
texture = "obj/models/spot/spot_texture.png"          # 可选
background_image = "backgrounds/skybox.jpg"           # 可选

[render]
width = 1920
height = 1080
projection = "perspective"                             # "perspective" | "orthographic"
use_zbuffer = true
use_texture = true
use_gamma = true
backface_culling = true
enhanced_ao = true                                     # 🔥 增强环境光遮蔽
soft_shadows = true                                    # 🔥 软阴影效果

[camera]
from = "2.5,1.5,4.0"                                  # 相机位置
at = "0,0.5,0"                                        # 观察目标
up = "0,1,0"                                          # 上方向
fov = 60.0                                            # 视场角(度)

[object]
position = "0,0.2,0"                                  # 物体位置
rotation = "15,30,0"                                  # 旋转角度(度)
scale_xyz = "1.2,1.0,1.2"                           # 非均匀缩放
scale = 1.5                                           # 全局缩放

[lighting]
use_lighting = true
ambient = 0.2                                         # 环境光强度
ambient_color = "0.2,0.3,0.4"                       # 环境光颜色

# 🔥 多光源配置 - 支持任意数量的光源
[[light]]
type = "directional"
enabled = true
direction = "0.3,-0.8,-0.5"
color = "1.0,0.95,0.8"
intensity = 0.8

[[light]]
type = "point"
enabled = true
position = "2.0,3.0,2.0"
color = "1.0,0.8,0.6"
intensity = 2.5
constant_attenuation = 1.0
linear_attenuation = 0.09
quadratic_attenuation = 0.032

[material]
use_phong = false                                     # Phong着色
use_pbr = true                                        # 🔥 推荐使用PBR

# === 🎨 Phong 增强参数 ===
diffuse_color = "0.7,0.5,0.3"                       # 漫反射颜色
diffuse_intensity = 1.2                             # 🔥 漫反射强度 (0.0-2.0)
specular_color = "0.9,0.8,0.7"                      # 🔥 镜面反射颜色 (RGB)
specular_intensity = 0.8                            # 🔥 镜面反射强度 (0.0-2.0)
shininess = 64.0                                     # 光泽度

# === 🔬 高级 PBR 参数 ===
base_color = "0.85,0.7,0.6"                         # 基础颜色
metallic = 0.0                                       # 金属度 (0.0-1.0)
roughness = 0.6                                      # 粗糙度 (0.0-1.0)
ambient_occlusion = 0.8                              # 环境光遮蔽

# 🔥 v2.2 新增高级效果
subsurface = 0.7                                     # ✨ 次表面散射强度 (0.0-1.0)
anisotropy = 0.0                                     # 🌟 各向异性 (-1.0 到 1.0)
normal_intensity = 0.8                               # 📐 法线强度 (0.0-2.0)
emissive = "0.0,0.0,0.0"                            # 自发光颜色

# === 阴影和环境光遮蔽设置 ===
enhanced_ao = true
ao_strength = 0.6
soft_shadows = true
shadow_strength = 0.8

[background]
use_background_image = false
enable_gradient_background = true                      # 渐变背景
gradient_top_color = "0.3,0.5,0.8"
gradient_bottom_color = "0.8,0.6,0.4"
enable_ground_plane = true                            # 地面平面
ground_plane_color = "0.4,0.6,0.3"
ground_plane_height = -0.5

[animation]
animate = false                                       # CLI动画模式
fps = 60                                             # 视频帧率
rotation_speed = 0.8                                 # 实时渲染速度
rotation_cycles = 2.0                                # 视频旋转圈数
animation_type = "CameraOrbit"                       # "CameraOrbit" | "ObjectLocalRotation" | "None"
rotation_axis = "Y"                                  # "X" | "Y" | "Z" | "Custom"
custom_rotation_axis = "0.2,1,0.3"                  # 自定义轴(当rotation_axis="Custom")
```

## 材质系统详解

### 🎨 Phong 增强着色模型

v2.2版本大幅增强了Phong着色模型，提供更精细的控制：

```toml
[material]
use_phong = true

# 🔥 独立强度控制
diffuse_color = "0.8,0.6,0.4"                       # 漫反射基色
diffuse_intensity = 1.5                             # 漫反射增强 50%

specular_color = "1.0,0.9,0.8"                      # 镜面反射色调
specular_intensity = 0.6                            # 降低镜面反射 40%

shininess = 128.0                                    # 高光泽度（锐利高光）
```

**Phong参数详解**：

| 参数 | 范围 | 效果描述 |
|------|------|----------|
| `diffuse_intensity` | 0.0-2.0 | **漫反射强度**，控制表面亮度 |
| `specular_intensity` | 0.0-2.0 | **镜面反射强度**，控制高光亮度 |
| `shininess` | 1.0-512.0 | **光泽度**，值越高高光越锐利 |
| `diffuse_color` | RGB | **漫反射颜色**，物体主要颜色 |
| `specular_color` | RGB | **🔥 镜面反射颜色**，高光色调 |

### 🔬 高级 PBR 材质系统

v2.2版本引入了业界领先的PBR参数：

```toml
[material]
use_pbr = true

# 基础PBR三要素
base_color = "0.8,0.6,0.4"                          # 基础反射率
metallic = 0.2                                       # 轻微金属感
roughness = 0.3                                      # 中等粗糙度

# 🔥 v2.2 高级效果
subsurface = 0.8                                     # ✨ 强次表面散射
anisotropy = 0.4                                     # 🌟 方向性反射
normal_intensity = 1.2                               # 📐 增强表面细节
```

#### ✨ 次表面散射 (Subsurface Scattering)

模拟光线在材质内部的散射效果，适用于：

```toml
# 🧑 人体皮肤效果
subsurface = 0.7
base_color = "0.9,0.7,0.6"
metallic = 0.0
roughness = 0.4

# 🕯️ 蜡烛材质
subsurface = 0.9
base_color = "0.95,0.9,0.8"
metallic = 0.0
roughness = 0.6

# 🏛️ 大理石效果
subsurface = 0.5
base_color = "0.9,0.9,0.85"
metallic = 0.1
roughness = 0.2
```

#### 🌟 各向异性 (Anisotropy)

控制表面反射的方向性特征：

```toml
# 🔧 拉丝金属效果
anisotropy = 0.8                                     # 强方向性
metallic = 0.9
roughness = 0.3
base_color = "0.8,0.8,0.9"

# 🧵 织物材质
anisotropy = -0.4                                    # 垂直方向
metallic = 0.0
roughness = 0.7
base_color = "0.6,0.4,0.3"

# 💿 CD光盘效果
anisotropy = 0.95                                    # 极强径向反射
metallic = 0.8
roughness = 0.1
```

#### 📐 法线强度 (Normal Intensity)

精细控制表面细节的强度：

```toml
# 🏔️ 粗糙岩石
normal_intensity = 1.8                               # 增强凹凸感
roughness = 0.9

# 🪞 光滑表面
normal_intensity = 0.3                               # 减弱表面变化
roughness = 0.1

# 🎭 程序化细节
normal_intensity = 1.5                               # 适中细节
# 当没有法线贴图时，会生成程序化表面变化
```

### 🎯 材质预设示例

#### 金属材质

```toml
# 🥇 抛光金属
[material]
use_pbr = true
base_color = "1.0,0.8,0.4"                          # 金色
metallic = 0.9                                       # 高金属度
roughness = 0.1                                      # 镜面光滑
anisotropy = 0.0                                     # 各向同性
subsurface = 0.0                                     # 无次表面散射

# 🔩 拉丝不锈钢
[material]
use_pbr = true
base_color = "0.8,0.8,0.9"                          # 冷色金属
metallic = 0.8
roughness = 0.3
anisotropy = 0.6                                     # 拉丝效果
normal_intensity = 1.2                               # 增强纹理
```

#### 有机材质

```toml
# 🍎 水果皮肤
[material]
use_pbr = true
base_color = "0.8,0.2,0.1"                          # 苹果红
metallic = 0.0                                       # 非金属
roughness = 0.4                                      # 半光滑
subsurface = 0.6                                     # 明显次表面散射
normal_intensity = 0.8                               # 适中表面纹理

# 🧑 人体皮肤
[material]
use_pbr = true
base_color = "0.9,0.7,0.6"                          # 肤色
metallic = 0.0
roughness = 0.5
subsurface = 0.8                                     # 强次表面散射
anisotropy = 0.0                                     # 各向同性
```

#### 特殊效果

```toml
# ✨ 发光材质
[material]
use_pbr = true
base_color = "0.2,0.4,0.8"                          # 基础蓝色
emissive = "0.1,0.3,0.6"                            # 蓝色发光
metallic = 0.0
roughness = 0.3
subsurface = 0.4                                     # 轻微内部发光

# 🌟 全息材质
[material]
use_pbr = true
base_color = "0.8,0.9,1.0"                          # 冷色基调
metallic = 0.6                                       # 部分金属性
roughness = 0.2                                      # 光滑
anisotropy = 0.8                                     # 强方向性
normal_intensity = 1.5                               # 增强细节
```

## GUI使用指南

### 界面布局

```
┌─────────────────────────────────────────────────────────────┐
│ 🏠 光栅化渲染器 | 状态信息 | FPS显示 | Ctrl+R: 快速渲染    │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│ 🎛️ 控制面板    │           🖼️ 渲染结果显示区域                │
│              │                                              │
│ 📁 文件设置     │           🖱️ 相机交互区域                   │
│ 🎨 渲染设置     │                                              │
│ 🔧 物体变换     │           右下角: 交互提示面板               │
│ 📷 相机设置     │                                              │
│ 💡 光照设置     │                                              │
│ 🎭 材质设置     │  ← 🔥 增强的材质面板                        │
│ 🎬 动画设置     │                                              │
│ 🔴 渲染按钮     │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

### 🎭 增强的材质设置面板

v2.2版本的材质面板提供直观的参数控制：

```
┌─────────────────────────────────────────┐
│ 🎭 材质与光照设置                        │
├─────────────────────────────────────────┤
│ 着色模型: ○ Phong  ● PBR               │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ � PBR 高级参数                         │
│ 基础颜色: [■] [0.85, 0.70, 0.60]       │
│ 金属度:   [████████░░] 0.0             │
│ 粗糙度:   [██████░░░░] 0.6             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ ✨ 次表面散射: [███████░░░] 0.7         │
│ 🌟 各向异性:   [█████░░░░░] 0.0         │
│ 📐 法线强度:   [████████░░] 0.8         │
│ 💡 自发光:     [■] [0.0, 0.0, 0.0]     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ 💡 提示: 次表面散射适用于皮肤、蜡烛等   │
└─────────────────────────────────────────┘
```

**材质参数实时预览**：

- 🎚️ **滑块调节** - 拖动滑块实时看到效果变化
- 🎨 **颜色选择器** - 点击色块打开颜色选择面板
- 💡 **参数提示** - 鼠标悬停显示参数详细说明
- 🔄 **实时更新** - 参数变化立即反映到渲染结果

### �🔥 配置文件管理

在"文件与输出设置"面板中：

- **📁 加载配置** - 从.toml文件加载完整配置
- **💾 保存配置** - 将当前设置保存为.toml文件
- **📋 示例配置** - 快速应用内置示例配置

```
┌─────────────────────────────────────────┐
│ 文件与输出设置                            │
├─────────────────────────────────────────┤
│ OBJ文件: [path/to/model.obj] [浏览...]   │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ 配置文件: [📁加载配置] [💾保存配置] [📋示例] │
│ 💡 提示：加载配置会覆盖当前所有设置       │
└─────────────────────────────────────────┘
```

### 🖱️ 相机交互系统

在中央渲染区域进行3D导航：

| 操作 | 功能 |
|------|------|
| **鼠标拖拽** | 平移相机视角 |
| **Shift + 拖拽** | 围绕目标轨道旋转 |
| **鼠标滚轮** | 推拉缩放 |
| **R键** | 重置到默认视角 |
| **F键** | 聚焦到物体中心 |

**敏感度调节**: 在"相机设置"面板中可独立调整平移、旋转、缩放的响应速度。

### 🎬 动画渲染模式

#### 实时动画渲染

```
[开始动画渲染] 按钮 → 立即开始旋转动画
├── 显示实时FPS统计
├── 支持相机交互调整观察角度  
├── 可调节旋转速度
└── [停止动画渲染] 停止播放
```

#### 预渲染模式

```
☑️ 启用预渲染模式 → [开始动画渲染]
├── 首次: 预先计算所有帧 (显示进度)
├── 完成后: 流畅播放预渲染帧
├── 适合复杂场景和高质量预览
└── 占用更多内存，但播放无卡顿
```

### 🎥 视频生成工作流

1. **配置参数**: 在"动画设置"中调整fps、旋转圈数、动画类型
2. **预览效果**: 使用实时动画渲染预览效果
3. **生成视频**: 点击"生成视频"按钮，后台渲染并合成MP4
4. **进度监控**: 状态栏显示渲染进度和预计时间

## 命令行模式

### CLI 参数总览

```bash
cargo run --release -- [OPTIONS]

OPTIONS:
    -c, --config <FILE>        📁 指定TOML配置文件路径
        --headless             🚀 无头模式(不启动GUI)
        --use-example-config   📋 使用示例配置
    -h, --help                 显示帮助信息
```

### 使用场景

#### 🔧 配置文件开发

```bash
# 1. 生成示例配置
cargo run --release -- --use-example-config

# 2. 编辑配置文件
notepad temp_example_config.toml  # Windows
# 或
vim temp_example_config.toml      # Linux/macOS

# 3. 测试配置 (GUI模式)
cargo run --release -- --config temp_example_config.toml

# 4. 无头批量渲染
cargo run --release -- --config temp_example_config.toml --headless
```

#### 🤖 自动化渲染

```bash
# 批量处理多个配置
for config in configs/*.toml; do
    cargo run --release -- --config "$config" --headless
done
```

## 渲染管线

```mermaid
graph TD
    A[TOML配置加载] --> B[场景构建]
    B --> C[几何变换管线]
    
    subgraph 几何处理
        C --> D[模型变换]
        D --> E[视图变换]
        E --> F[投影变换]
        F --> G[视口变换]
    end
    
    G --> H[三角形筛选]
    
    subgraph 智能剔除
        H --> H1[视锥剔除]
        H1 --> H2[背面剔除]
        H2 --> H3[小三角形剔除]
    end
    
    H3 --> I[多线程光栅化]
    
    subgraph 像素着色
        I --> J1[重心坐标插值]
        J1 --> J2[深度测试]
        J2 --> J3[纹理采样]
        J3 --> J4[法线插值]
    end
    
    subgraph 高级光照
        J4 --> K1{着色模型}
        K1 -->|Phong| K2[🔥增强Blinn-Phong]
        K1 -->|PBR| K3[🔬高级物理渲染]
        K2 --> K4[多光源计算]
        K3 --> K5[✨次表面散射]
        K5 --> K6[🌟各向异性反射]
        K6 --> K7[📐法线强度处理]
        K4 --> K8[🔥 增强AO]
        K7 --> K8
        K8 --> K9[🔥 软阴影]
        K9 --> K10[Gamma校正]
    end
    
    K10 --> L[帧缓冲输出]
    L --> M[图像保存/显示]
    
    subgraph 动画系统
        M --> N1[实时动画]
        M --> N2[预渲染帧序列]
        N2 --> N3[FFmpeg视频合成]
    end
    
    subgraph GUI交互
        M --> O1[相机交互]
        O1 --> O2[参数实时调整]
        O2 --> C
    end
```

## 项目架构

```
src/
├── 🏗️ core/                    # 核心渲染引擎
│   ├── frame_buffer.rs         # 帧缓冲区与背景管理
│   ├── geometry_processor.rs   # 几何变换处理器
│   ├── renderer.rs            # 主渲染器协调
│   ├── triangle_processor.rs  # 三角形处理与准备
│   ├── parallel_rasterizer.rs # 🔥 智能并行光栅化器
│   ├── rasterizer/            # 🔥 模块化光栅化系统
│   │   ├── mod.rs             # 模块导出
│   │   ├── triangle_data.rs   # 核心数据结构
│   │   ├── pixel_processor.rs # 像素处理核心
│   │   ├── color_calculator.rs# 颜色计算算法
│   │   ├── texture_sampler.rs # 纹理采样系统
│   │   └── lighting_effects.rs# 光照效果计算
│   └── mod.rs                 # 核心模块导出
├── 📐 geometry/                # 几何处理模块  
│   ├── camera.rs              # 专业相机系统
│   ├── transform.rs           # 几何变换矩阵
│   ├── culling.rs             # 智能剔除算法
│   └── interpolation.rs       # 插值算法
├── 📁 io/                      # 🔥 配置与IO系统
│   ├── config_loader.rs       # TOML配置管理器
│   ├── simple_cli.rs          # 极简CLI处理
│   ├── render_settings.rs     # 统一配置数据结构
│   └── obj_loader.rs          # 模型资源加载
├── 💡 material_system/         # 🔬 增强材质与光照
│   ├── light.rs               # 多光源系统
│   ├── materials.rs           # 🔥 Phong/PBR增强材质
│   ├── texture.rs             # 纹理管理
│   └── color.rs               # 颜色处理
├── 🎬 scene/                   # 场景管理
│   ├── scene_utils.rs         # 场景构建与统计
│   └── scene_object.rs        # 场景对象变换
├── 🖥️ ui/                      # 现代化GUI界面
│   ├── app.rs                 # eframe应用主逻辑
│   ├── widgets.rs             # 自定义UI组件
│   ├── render_ui.rs           # 文件选择与配置管理
│   ├── core.rs                # GUI核心方法
│   └── animation.rs           # 动画控制逻辑
├── 🛠️ utils/                   # 工具函数库
│   ├── render_utils.rs        # 渲染辅助函数
│   ├── model_utils.rs         # 模型处理工具
│   └── save_utils.rs          # 文件保存工具
└── main.rs                    # 程序入口点
```

### 🔥 v2.2 架构亮点

- **🎯 配置驱动架构**: `RenderSettings`作为单一数据源，CLI/GUI/TOML三者统一
- **� 增强材质系统**: 次表面散射、各向异性、法线强度等高级PBR特性
- **�📦 模块化设计**: 光栅化器拆分为多个专职模块，便于维护和扩展
- **⚡ 智能并行渲染**: 自动选择最优并行策略，无需手动配置
- **🔄 实时交互**: GUI参数变化立即反映到渲染结果
- **💾 状态管理**: 统一的错误处理、进度监控、资源清理

## 示例与教程

### 📚 材质效果示例

#### 次表面散射演示

```bash
# 1. 创建皮肤材质配置
cargo run --release -- --use-example-config

# 2. 编辑配置实现皮肤效果
[material]
use_pbr = true
base_color = "0.9,0.7,0.6"      # 肤色
metallic = 0.0                   # 非金属
roughness = 0.4                  # 半光滑
subsurface = 0.8                 # 🔥 强次表面散射
normal_intensity = 0.8           # 适中表面细节

# 3. GUI模式查看效果
cargo run --release -- --config temp_example_config.toml
```

#### 各向异性金属效果

```toml
[material]
use_pbr = true
base_color = "0.8,0.8,0.9"       # 不锈钢色
metallic = 0.85                  # 高金属度
roughness = 0.3                  # 中等粗糙度
anisotropy = 0.7                 # 🔥 强方向性（拉丝效果）
normal_intensity = 1.2           # 增强表面纹理

[[light]]
type = "directional"
direction = "0.3,-0.8,-0.5"
intensity = 1.0

[[light]]
type = "point"                   # 添加侧光展示各向异性
position = "2.0,1.0,0.0"
intensity = 0.6
```

#### 发光材质演示

```toml
[material]
use_pbr = true
base_color = "0.1,0.3,0.8"       # 深蓝基色
emissive = "0.2,0.4,1.0"         # 🔥 蓝色发光
metallic = 0.0
roughness = 0.2
subsurface = 0.3                 # 轻微内部散射增强发光感

# 调暗环境光突出发光效果
[lighting]
ambient = 0.1
ambient_color = "0.1,0.1,0.2"
```

### 🎬 动画制作工作流

#### 材质动画展示

```toml
[animation]
animation_type = "CameraOrbit"
rotation_axis = "Y"
rotation_cycles = 2.0            # 两圈展示
fps = 30

# 使用高对比度光照展示材质效果
[[light]]
type = "directional"
direction = "0.5,-1.0,-0.3"
color = "1.0,0.9,0.8"            # 暖色主光
intensity = 1.2

[[light]]
type = "point"
position = "3.0,2.0,3.0"
color = "0.6,0.8,1.0"            # 冷色补光
intensity = 0.8
```

**操作流程**:

1. 在GUI中设置材质参数
2. 点击"开始动画渲染"预览旋转效果
3. 观察光照条件下的材质表现
4. 调整参数获得最佳效果
5. 生成高质量视频展示

### 🏭 生产环境配置

#### 高质量材质渲染

```toml
[render]
width = 2560                     # 2.5K分辨率
height = 1440
enhanced_ao = true
ao_strength = 0.8
soft_shadows = true
shadow_strength = 0.6

[material]
use_pbr = true
# 高质量次表面散射设置
subsurface = 0.7
normal_intensity = 1.0           # 保持适中避免过度
ambient_occlusion = 0.9          # 高环境光遮蔽

# 专业三点布光
[[light]]
type = "directional"             # 主光源
direction = "0.3,-0.8,-0.5"
intensity = 1.0
color = "1.0,0.95,0.9"

[[light]]
type = "point"                   # 补光源
position = "-2.0,1.5,2.0"
intensity = 0.6
color = "0.9,0.95,1.0"

[[light]]
type = "point"                   # 轮廓光
position = "1.5,0.5,-2.0"
intensity = 0.4
color = "1.0,0.9,0.8"
```

#### 批量材质测试

```bash
#!/bin/bash
# material_test.sh - 批量测试不同材质效果

materials=("metal" "skin" "plastic" "fabric")
subsurface_values=(0.0 0.3 0.6 0.9)
anisotropy_values=(0.0 0.4 0.8)

for material in "${materials[@]}"; do
    for subsurface in "${subsurface_values[@]}"; do
        for anisotropy in "${anisotropy_values[@]}"; do
            config_file="test_${material}_s${subsurface}_a${anisotropy}.toml"
            
            # 生成配置文件
            cat > "$config_file" << EOF
[material]
use_pbr = true
subsurface = $subsurface
anisotropy = $anisotropy
# ... 其他材质参数
EOF
            
            # 渲染
            cargo run --release -- --config "$config_file" --headless
            
            echo "✅ 完成: $material (subsurface=$subsurface, anisotropy=$anisotropy)"
        done
    done
done
```

## 性能优化建议

### 💻 硬件配置

- **CPU**: 推荐8核心以上，受益于多线程光栅化
- **内存**: 16GB+，支持大模型和复杂材质计算
- **存储**: SSD优先，加速纹理和模型加载

### ⚙️ 渲染设置优化

#### 高性能设置

```toml
[render]
cull_small_triangles = true      # 剔除小三角形
min_triangle_area = 0.001       # 剔除阈值
backface_culling = true         # 背面剔除

[material]
# 简化材质计算
subsurface = 0.0                # 禁用次表面散射
anisotropy = 0.0                # 禁用各向异性
normal_intensity = 1.0          # 标准法线强度
```

#### 高质量设置

```toml
[render]
enhanced_ao = true
ao_strength = 0.8
soft_shadows = true
shadow_strength = 0.6

[material]
# 全功能材质
subsurface = 0.6                # 启用次表面散射
anisotropy = 0.4                # 适度各向异性
normal_intensity = 1.2          # 增强细节
```

#### 平衡设置

```toml
[render]
enhanced_ao = true
ao_strength = 0.5               # 适中AO强度
soft_shadows = true  
shadow_strength = 0.4           # 适中阴影强度

[material]
subsurface = 0.3                # 轻微次表面散射
anisotropy = 0.0                # 禁用各向异性（节省计算）
normal_intensity = 1.0          # 标准细节
```

## 故障排除

### 常见问题

#### 🔧 编译问题

```bash
# 确保Rust版本
rustc --version  # 应为1.81+

# 清理重新构建
cargo clean
cargo build --release
```

#### 🎨 材质效果不明显

```bash
# 检查光照设置
# 次表面散射需要适当的光照才能显现
[[light]]
type = "directional"
intensity = 1.0                 # 确保足够亮度

# 各向异性需要侧光源
[[light]]
type = "point"
position = "2.0,0.0,0.0"        # 侧面光源
```

#### 📁 文件加载问题

```bash
# 检查文件路径
ls -la obj/models/        # Linux/macOS
dir obj\models\           # Windows

# 验证文件格式
file model.obj            # 应显示ASCII text
```

#### 🎥 视频生成问题

```bash
# 检查FFmpeg
ffmpeg -version

# Windows安装FFmpeg
choco install ffmpeg

# 检查输出目录权限
ls -la output_directory/
```

### 📊 性能分析

#### 材质计算性能

```toml
# 性能测试配置
[material]
# 测试1: 基础PBR
use_pbr = true
subsurface = 0.0
anisotropy = 0.0

# 测试2: 次表面散射
subsurface = 0.5
anisotropy = 0.0

# 测试3: 各向异性
subsurface = 0.0
anisotropy = 0.5

# 测试4: 全功能
subsurface = 0.5
anisotropy = 0.5
```

在GUI中对比FPS差异，选择合适的质量/性能平衡点。

## 贡献指南

### 🛠️ 开发环境搭建

```bash
# 1. Fork项目
git clone https://github.com/Rukkhadevata123/Rasterizer_rust
cd Rasterizer_rust

# 2. 创建开发分支
git checkout -b feature/material-enhancement

# 3. 安装开发依赖
cargo install cargo-clippy cargo-fmt

# 4. 运行测试
cargo test
cargo clippy
cargo fmt
```

### 📝 代码规范

- **Rust风格**: 遵循官方Rust代码规范
- **注释语言**: 中文注释，英文变量名
- **提交信息**: 使用约定式提交格式
- **测试覆盖**: 新功能需要相应测试

### 🎨 材质系统扩展

贡献新材质效果时，请遵循以下结构：

```rust
// src/material_system/materials.rs

pub mod pbr_functions {
    // 新增材质函数应放在这里
    pub fn your_new_effect(/* 参数 */) -> Vector3<f32> {
        // 实现新效果
    }
}

// 在 MaterialView::compute_response 中集成
// 在 RenderSettings 中添加对应参数
// 在 GUI 中添加控制滑块
```

## 版本历史

### 🎉 v2.2.0 (Current)

- ✨ **增强PBR材质系统**
  - 🔥 次表面散射效果
  - 🌟 各向异性反射
  - 📐 可调法线强度
- 🎨 **Phong着色增强**
  - 独立漫反射/镜面反射强度控制
  - 镜面反射颜色支持
- 🖥️ **材质GUI优化**
  - 直观的参数滑块
  - 实时效果预览
  - 材质预设支持

### 🎉 v2.0.0

- ✨ **全新TOML配置系统**
- 🖥️ **现代化GUI界面重构**  
- 📁 **配置文件管理功能**
- 🔥 **增强AO和软阴影算法**
- 🎬 **预渲染动画系统**
- 🖱️ **专业级相机交互**

### v1.x Legacy

- 基础光栅化渲染器
- CLI参数配置
- 基础Phong/PBR着色模型
- 多线程渲染支持

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- **egui**: 现代化Rust GUI框架
- **nalgebra**: 高性能线性代数库
- **image**: 图像处理库
- **toml**: TOML配置解析
- **clap**: CLI参数解析

---

<div align="center">

**🎨 用Rust重新定义软件光栅化渲染 🎨**

**🔬 v2.2: 业界领先的材质系统，专业级渲染效果 🔬**

[🔗 GitHub仓库](https://github.com/Rukkhadevata123/Rasterizer_rust) | [📚 文档](README.md) | [🐛 问题反馈](https://github.com/Rukkhadevata123/Rasterizer_rust/issues)

</div>
