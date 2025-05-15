{
  description = "OpenGL/Vulkan 图形开发环境";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
  in {
    devShells."${system}".default = let
      pkgs = import nixpkgs {
        inherit system;
        config = {allowUnfree = true;};
      };
    in
      pkgs.mkShell {
        buildInputs = with pkgs; [
          # 构建工具
          cmake
          gnumake
          ninja
          pkg-config
          shaderc
          glslang
          stb

          # 添加 ASSIMP 库
          assimp # Open Asset Import Library

          # GLFW及其依赖
          glfw3 # GLFW 库本身
          xorg.libX11 # X11 库
          xorg.libXrandr
          xorg.libXinerama
          xorg.libXcursor
          xorg.libXi
          xorg.libXext
          vulkan-loader # Vulkan 支持
          vulkan-headers
          vulkan-tools
          vulkan-validation-layers
          spirv-tools
          spirv-headers

          tinyobjloader # TinyObjLoader 库

          # 常见图形库和依赖
          libGL.dev
          libGLU.dev
          freeglut
          mesa
          SDL2
          sdl3
          sdl3-ttf
          sdl3-image

          # 对于 Wayland（如果需要）
          wayland
          wayland-scanner
          wayland-protocols
          libxkbcommon

          # 添加 FreeType 库
          freetype
          fontconfig

          # 图像处理库
          libpng
          libjpeg

          # 其他可能需要的依赖
          xorg.libXxf86vm
          libffi
          glm # OpenGL Mathematics
        ];

        # 设置环境变量
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
            # 图形库
            pkgs.assimp
            pkgs.freetype
            pkgs.fontconfig
            pkgs.libpng
            pkgs.libjpeg
            pkgs.glfw3
            pkgs.xorg.libX11
            pkgs.libGL
            pkgs.vulkan-loader
            pkgs.libGLU
            pkgs.freeglut
            pkgs.mesa
            pkgs.wayland
            pkgs.libxkbcommon
            pkgs.xorg.libXxf86vm
            pkgs.libffi
            pkgs.SDL2
            pkgs.sdl3
            pkgs.sdl3-ttf
            pkgs.sdl3-image
          ]}"

          # STB 头文件路径设置
          export STB_INCLUDE_PATH="${pkgs.stb}/include/stb"
          export C_INCLUDE_PATH="${pkgs.glfw3}/include:${pkgs.vulkan-headers}/include:${pkgs.libGL.dev}/include:${pkgs.libGLU.dev}/include:${pkgs.stb}/include:${pkgs.mesa}/include:$C_INCLUDE_PATH"
          export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"

          # Vulkan 验证层设置
          export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
          export XDG_DATA_DIRS="$XDG_DATA_DIRS:${pkgs.vulkan-validation-layers}/share"

          # 简化的 Vulkan ICD 设置 - 使用系统提供的路径
          if [ -d "/run/opengl-driver/share/vulkan/icd.d" ]; then
            # 收集 64 位和 32 位 ICD 文件
            ICD_FILES=$(find /run/opengl-driver{,-32}/share/vulkan/icd.d -name "*.json" 2>/dev/null | tr '\n' ':')

            if [ ! -z "$ICD_FILES" ]; then
              # 移除末尾的冒号
              ICD_FILES="''${ICD_FILES%:}"
              export VK_ICD_FILENAMES="$ICD_FILES"
              echo "已设置 VK_ICD_FILENAMES = $VK_ICD_FILENAMES"
            else
              # 回退到软件渲染器
              if [ -f "${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json" ]; then
                export VK_ICD_FILENAMES="${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json"
                echo "未找到硬件 ICD，使用软件渲染器: $VK_ICD_FILENAMES"
              else
                echo "警告: 未找到任何 Vulkan ICD 文件！"
              fi
            fi
          else
            echo "警告: /run/opengl-driver 目录不存在，可能缺少驱动"
          fi

          # CMake 辅助设置
          export ASSIMP_INCLUDE_DIR="${pkgs.assimp}/include"
          export ASSIMP_LIBRARY="${pkgs.assimp}/lib/libassimp.so"
          export FREETYPE_INCLUDE_DIRS="${pkgs.freetype.dev}/include/freetype2"
          export FREETYPE_LIBRARY="${pkgs.freetype}/lib/libfreetype.so"
          export GLFW3_INCLUDE_DIR="${pkgs.glfw3}/include"
          export GLFW3_LIBRARY="${pkgs.glfw3}/lib/libglfw.so"
          export OpenGL_GL_PREFERENCE="GLVND"
          export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:${pkgs.glfw3}:${pkgs.assimp}:${pkgs.freetype}:${pkgs.fontconfig}:${pkgs.stb}"

          # 创建 CMake 工具链文件
          cat > opengl_toolchain.cmake << EOF
          set(OpenGL_GL_PREFERENCE "GLVND")
          if(POLICY CMP0072)
            cmake_policy(SET CMP0072 NEW)
          endif()
          set(STB_INCLUDE_DIR "${pkgs.stb}/include")
          EOF

          # 创建简化的 Vulkan 检查脚本
          cat > check_vulkan.sh << EOF
          #!/bin/sh
          echo "==== Vulkan 系统信息 ===="
          echo "VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
          echo "VK_LAYER_PATH: $VK_LAYER_PATH"

          # 系统 ICD 文件
          echo -e "\n==== 系统 ICD 文件 ===="
          for d in /run/opengl-driver/share/vulkan/icd.d /run/opengl-driver-32/share/vulkan/icd.d; do
            if [ -d "\$d" ]; then
              echo "\$d:"
              ls -la "\$d"
            fi
          done

          # 运行 vulkaninfo
          if command -v vulkaninfo &> /dev/null; then
            echo -e "\n==== Vulkan 设备信息 ===="
            vulkaninfo --summary 2>/dev/null || echo "vulkaninfo 执行失败"
          fi
          EOF
          chmod +x check_vulkan.sh

          echo "OpenGL/Vulkan 开发环境已准备就绪"
          echo "要诊断 Vulkan 问题，请运行: ./check_vulkan.sh"
        '';
      };
  };
}
