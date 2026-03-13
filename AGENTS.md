# Rasterizer Rust - 项目上下文文档

## 项目概述

这是一个用 Rust 编写的高性能、多线程软件光栅化渲染器，从零实现，不依赖 GPU API。该项目实现了现代可编程渲染管线，具备以下核心特性：

### 核心功能
- **PBR（基于物理的渲染）**：采用标准 Metallic-Roughness 工作流
  - Cook-Torrance 镜面 BRDF，使用 Trowbridge-Reitz GGX 分布和 Smith 几何函数
  - Fresnel-Schlick 近似处理不同角度的真实光反射
  - 完整支持切线空间法线映射（MikkTSpace 兼容的切线生成）
  - ACES 电影色调映射，实现电影级色彩重现

- **高级渲染能力**
  - **透明度**：通过从后到前排序和 Alpha 混合逻辑正确渲染半透明物体（玻璃、冰）
  - **阴影映射**：使用 PCF（百分比逼近过滤）实现实时软阴影，自适应阴影偏差消除伪影
  - **抗锯齿**：SSAA（超级采样抗锯齿）支持，边缘平滑
  - **纹理过滤**：三线性过滤 + Mipmap 支持，高质量纹理采样，无混叠

- **高性能架构**
  - **大规模并行化**：利用 `Rayon` 并行化顶点处理、三角形光栅化、片段着色和后处理步骤
  - **线程安全**：自定义 `FrameBuffer` 使用原子深度缓冲区和条带锁机制，实现无竞争并发写入
  - **裁剪**：健壮的齐次裁剪空间裁剪（Sutherland-Hodgman），正确处理相机平面附近的图元
  - **优化**：数据导向设计，针对复杂场景排序进行预变换优化

- **交互式实时 GUI**
  - **窗口管理**：通过 `minifb` 实现轻量级零开销窗口
  - **相机**：FPS 风格自由漫游相机，WASD 移动 + 鼠标视角
  - **热重载**：按 'R' 键即时重载场景配置，无需重启应用
  - **运行时工具**：实时切换线框模式（中键）和裁剪模式（右键）

### 项目技术栈
- **语言**：Rust 2024 edition
- **数学库**：nalgebra 0.34
- **并行处理**：rayon 1.11
- **3D 资产加载**：gltf 1.0
- **窗口系统**：minifb 0.28
- **图像处理**：image 0.25
- **配置序列化**：serde + toml 0.9
- **命令行**：clap 4.5
- **日志**：env_logger + log 0.4

## 项目架构

项目采用清晰的模块化架构，将核心引擎逻辑与场景管理和管线定义分离：

```
src/
├── core/              # 引擎内核
│   ├── rasterizer.rs  # 扫描线光栅化与裁剪逻辑
│   ├── framebuffer.rs # 线程安全原子缓冲区管理
│   ├── geometry.rs    # 顶点布局与几何图元
│   └── math/          # 变换工厂与插值辅助
│       ├── interpolation.rs  # 重心坐标与透视正确插值
│       └── transform.rs      # 矩阵变换工具
├── pipeline/          # 渲染管线
│   ├── passes.rs      # 高级渲染通道（阴影与主渲染）
│   ├── renderer.rs    # 渲染编排器与清除逻辑
│   └── shaders/       # 可编程 PBR 与阴影着色器
│       ├── pbr.rs     # PBR 着色器实现
│       └── shadow.rs  # 阴影着色器
├── scene/             # 场景图与资源
│   ├── material.rs    # PBR 材质与 Alpha 模式定义
│   ├── texture.rs     # 纹理加载、Mipmap 生成与过滤
│   ├── light.rs       # 光照定义（方向光、点光源）
│   ├── loader.rs      # 资源管理与热重载
│   ├── mesh.rs        # 网格数据结构
│   ├── model.rs       # 模型容器
│   ├── camera.rs      # 相机定义
│   └── context.rs     # 渲染上下文
├── io/                # 文件 I/O
│   ├── gltf_loader.rs # 健壮的 glTF 2.0 资产导入器
│   ├── config.rs      # TOML 场景配置
│   └── image.rs       # 图像保存
├── ui/                # 用户界面
│   └── input.rs       # 输入控制与相机控制器
├── app.rs             # 应用程序控制循环（GUI/CLI 模式）
└── main.rs            # 入口点
```

## 构建和运行

### 前置要求
- Rust 最新稳定版
- Cargo

### 构建项目
```bash
# 开发构建
cargo build

# 发布构建（推荐用于渲染）
cargo build --release
```

### 运行项目

**1. 实时 GUI 模式（推荐）**
启动交互式查看器，实时探索场景、测试光照和查看 PBR 材质。

```bash
cargo run --release -- --config scene.toml --gui
```

**2. 离线渲染（CLI 模式）**
渲染单张高质量帧到输出图像文件（默认：`output.png`）。

```bash
cargo run --release -- --config scene.toml
```

### GUI 模式控制

| 输入 | 动作 |
|------|------|
| **W / A / S / D** | 移动相机 |
| **鼠标** | 环顾四周 |
| **空格 / 左 Shift** | 向上 / 向下移动 |
| **滚轮** | 调整 FOV（缩放） |
| **R** | 重新加载配置（热重载） |
| **右键** | 循环裁剪模式（背面 → 无 → 正面） |
| **中键** | 切换线框模式 |
| **Esc** | 退出应用 |

### 运行测试
```bash
cargo test
```

## 配置文件

项目使用 TOML 格式的配置文件（默认：`scene.toml`）来定义渲染参数、场景对象、光照和相机设置。

### 主要配置节

**[render] - 渲染设置**
- `width`, `height`: 输出分辨率
- `output`: 输出图像路径
- `samples`: MSAA 样本数
- `exposure`: 色调映射曝光
- `ambient_light`: 环境光颜色
- `background_gradient_*`: 渐变背景（上/下）
- `use_shadows`: 启用阴影映射
- `shadow_map_size`: 阴影贴图分辨率
- `shadow_ortho_size`: 方向光阴影的正交投影大小
- `shadow_bias`: 阴影偏差（防止阴影痤疮）
- `use_pcf`: 启用 PCF 软阴影
- `pcf_kernel_size`: PCF 采样半径
- `use_aces`: ACES 电影色调映射
- `cull_mode`: 裁剪模式（"back", "front", "none"）
- `use_mipmap`: 生成和使用纹理 Mipmap
- `wireframe`: 线框模式（调试）

**[camera] - 相机设置**
- `projection`: 投影类型（"perspective" 或 "orthographic"）
- `position`, `target`, `up`: 相机位置和朝向
- `fov`: 视场角（透视投影）
- `near`, `far`: 裁剪平面
- `speed`, `sensitivity`, `zoom_speed`: 控制参数

**[ground] - 地面设置**
- `enabled`: 启用地面
- `size`: 地面大小
- `albedo`, `metallic`, `roughness`: 地面材质属性

**[[lights]] - 光照列表**
- `type`: 光照类型（"directional" 或 "point"）
- `direction`: 方向光方向
- `position`: 点光源位置
- `color`: 光照颜色
- `intensity`: 光照强度

**[[objects]] - 对象列表**
- `path`: glTF/GLB 文件路径
- `position`: 位置
- `rotation`: 旋转角度（度）
- `scale`: 缩放

## 开发约定

### 代码风格
- 使用 Rust 2024 edition
- 遵循 Rust 官方代码风格指南
- 使用 `cargo fmt` 格式化代码
- 使用 `cargo clippy` 进行 lint 检查

### 命名约定
- 类型：`PascalCase`
- 函数/方法：`snake_case`
- 常量：`SCREAMING_SNAKE_CASE`
- 私有字段：`snake_case`（无下划线前缀）

### 模块组织
- 每个模块有单一职责
- 使用 `pub mod` 导出公开接口
- 实现细节保持私有

### 并发处理
- 使用 `rayon` 进行数据并行
- 原子操作用于共享状态（如深度缓冲区）
- 避免 `unsafe` 代码，必要时添加详细注释

### 错误处理
- 使用 `Result<T, String>` 用于可恢复错误
- 使用 `panic!` 仅用于不可恢复的致命错误
- 使用 `log` crate 进行日志记录

### 性能优化原则
- 优先考虑算法优化而非微优化
- 使用缓存避免重复计算
- 批量处理减少函数调用开销
- 预分配容器容量避免重新分配

### 测试策略
- 单元测试：核心数学和算法
- 集成测试：渲染管线和资产加载
- 基准测试：性能关键路径

### 资产管理
- glTF 2.0 作为主要 3D 格式
- 纹理支持：PNG、JPG
- 支持嵌套的 glTF 目录结构
- 热重载时缓存纹理以避免重复加载

## 关键实现细节

### 光栅化管线
1. **顶点处理**：并行处理顶点变换
2. **裁剪**：Sutherland-Hodgman 在齐次裁剪空间
3. **透视除法**：NDC 变换
4. **视口变换**：屏幕坐标
5. **背面剔除**：基于屏幕空间有符号面积
6. **扫描线光栅化**：并行像素处理
7. **深度测试**：原子深度缓冲区
8. **片段着色**：PBR 计算
9. **输出合并**：不透明/透明混合

### PBR 着色器
- **NDF**：GGX/Trowbridge-Reitz
- **几何函数**：Smith 方法（Schlick-GGX）
- **Fresnel**：Schlick 近似
- **法线映射**：切线空间到世界空间变换
- **IBL**：当前仅基础环境光（未来可扩展）

### 阴影映射
- 方向光使用正交投影
- PCF 软阴影
- 自适应偏差基于表面角度
- 仅对第一个光源应用阴影

### 透明度处理
- 背面到前面排序
- Alpha 混合：`src * a + dst * (1-a)`
- 不写入深度缓冲区
- 在不透明通道之后渲染

## 常见任务

### 添加新的着色器
1. 在 `src/pipeline/shaders/` 创建新文件
2. 实现 `Shader` trait
3. 定义 `Varying` 类型
4. 实现 `vertex` 和 `fragment` 方法
5. 在渲染通道中使用

### 加载新的 3D 模型
1. 将 `.gltf` 或 `.glb` 文件放入 `assets/glbs/` 或 `assets/glTFs/`
2. 在 `scene.toml` 中添加 `[[objects]]` 条目
3. 指定路径、位置、旋转、缩放
4. 运行时按 'R' 热重载

### 调整渲染质量
1. 修改 `scene.toml` 中的 `samples`（SSAA）
2. 调整 `shadow_map_size` 提高阴影质量
3. 启用 `use_pcf` 软阴影
4. 调整 `pcf_kernel_size` 控制阴影软度

### 性能分析
1. 使用 `cargo build --release` 构建优化版本
2. 监控 GUI 模式的 FPS
3. 检查日志中的渲染时间
4. 使用 `rayon` 的线程池配置调整并行度

## 已知限制

- 当前仅支持 glTF 2.0 格式
- IBL（基于图像的照明）未完全实现
- 无延迟渲染支持
- 动画/蒙皮未实现
- 实例化渲染未实现
- 纹理压缩格式有限

## 未来改进方向

- 完整 IBL 实现（环境贴图）
- 动画和蒙皮支持
- 延迟渲染管线
- 后处理效果（Bloom、DOF、运动模糊）
- 更多光照模型（次表面散射）
- 性能分析工具集成
- Vulkan/WebGPU 后端选项

## 相关资源

- **GitHub**: https://github.com/Rukkhadevata123/rasterizer-rust
- **文档**: DeepWiki 集成（见 README 徽章）
- **问题报告**: 使用 GitHub Issues

## 许可证

MIT License - 详见 LICENSE 文件