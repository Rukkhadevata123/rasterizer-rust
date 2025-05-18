#!/bin/sh
echo "==== Vulkan 系统信息 ===="
echo "VK_ICD_FILENAMES: /run/opengl-driver/share/vulkan/icd.d/dzn_icd.x86_64.json:/run/opengl-driver/share/vulkan/icd.d/intel_hasvk_icd.x86_64.json:/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json:/run/opengl-driver/share/vulkan/icd.d/lvp_icd.x86_64.json:/run/opengl-driver/share/vulkan/icd.d/nouveau_icd.x86_64.json:/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json:/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json:/run/opengl-driver/share/vulkan/icd.d/virtio_icd.x86_64.json:/run/opengl-driver-32/share/vulkan/icd.d/dzn_icd.i686.json:/run/opengl-driver-32/share/vulkan/icd.d/intel_hasvk_icd.i686.json:/run/opengl-driver-32/share/vulkan/icd.d/intel_icd.i686.json:/run/opengl-driver-32/share/vulkan/icd.d/lvp_icd.i686.json:/run/opengl-driver-32/share/vulkan/icd.d/nouveau_icd.i686.json:/run/opengl-driver-32/share/vulkan/icd.d/nvidia_icd.i686.json:/run/opengl-driver-32/share/vulkan/icd.d/radeon_icd.i686.json:/run/opengl-driver-32/share/vulkan/icd.d/virtio_icd.i686.json"
echo "VK_LAYER_PATH: /nix/store/6d2v2wap7jkk6wvl65cs7719b8mvzs8h-vulkan-validation-layers-1.4.313.0/share/vulkan/explicit_layer.d"

# 系统 ICD 文件
echo -e "\n==== 系统 ICD 文件 ===="
for d in /run/opengl-driver/share/vulkan/icd.d /run/opengl-driver-32/share/vulkan/icd.d; do
  if [ -d "$d" ]; then
    echo "$d:"
    ls -la "$d"
  fi
done

# 运行 vulkaninfo
if command -v vulkaninfo &> /dev/null; then
  echo -e "\n==== Vulkan 设备信息 ===="
  vulkaninfo --summary 2>/dev/null || echo "vulkaninfo 执行失败"
fi
