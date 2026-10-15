# ComfyUI-ColorMatchPlus

HULK666_Corrector 插件的功能等价替代节点包。原 HULK666_Corrector 无公开仓库，本包提供两个
与截图工作流功能等价的节点，节点分类沿用 `HULK666_Corrector`，可直接替换原工作流中的节点。

## 功能

### 1. 精确颜色匹配（ColorMatchPlus）

- 输入：`image`（IMAGE）
- 参数：
  - `exposure`：曝光，0.0 ~ 2.0，默认 1.00（线性相乘）
  - `contrast`：对比度，0.0 ~ 2.0，默认 1.00（(x-0.5)*c+0.5）
  - `color_temp`：色温，-1.0 ~ 1.0，默认 0.00（>0 偏暖，<0 偏冷，通道白平衡增益）
  - `tint`：色调，-1.0 ~ 1.0，默认 0.00（>0 偏品红，<0 偏绿）
  - `device`：auto / cpu / cuda
- 处理链：曝光（线性乘）→ 对比度 → 色温/色调（按通道白平衡偏移）
- 输出：`IMAGE`

### 2. 高光细节恢复（HighlightRestorePlus）

- 输入：`image`（IMAGE，已处理图）、`original_image`（IMAGE，细节来源）
- 参数：
  - `threshold`：阈值，0.0 ~ 1.0，默认 0.50（亮度掩码 luma > threshold 为高光区，平滑过渡）
  - `strength`：恢复强度，0.0 ~ 1.0，默认 0.00
  - `device`：auto / cpu / cuda
- 功能：将 `original_image` 的高频细节（原图减高斯模糊）按亮度掩码与 `strength`
  融合回 `image`，恢复高光区域被压缩掉的细节。
- 输出：`IMAGE`

## 特性

- 纯 torch/numpy 实现，不引入任何第三方新依赖，不升降级核心包
- 支持 batch 输入
- 支持 alpha 通道透传（输入含 4 通道时原样保留 alpha）

## 安装

将本目录（ComfyUI-ColorMatchPlus）复制到 ComfyUI 的 `custom_nodes` 目录下，重启
ComfyUI 即可。节点分类 `HULK666_Corrector`，无需额外安装依赖。

## 版本

- v1.0.1：节点分类名改为 HULK666_Corrector（原分类弃用），中文节点显示名与功能不变。
- v1.0.0：初版，实现上述两节点。