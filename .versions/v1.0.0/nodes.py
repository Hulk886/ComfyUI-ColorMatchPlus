import torch
import torch.nn.functional as F

# ============================================================
# ComfyUI-ColorMatchPlus
#   与 JINGSI_Corrector 功能等价的节点包（分类沿用 JINGSI_Corrector）
#   - ColorMatchPlus        精确颜色匹配
#   - HighlightRestorePlus  高光细节恢复
# 纯 torch 实现，不引入任何第三方新依赖；
# 支持 batch 输入与 alpha 通道透传；device: auto/cpu/cuda
# ============================================================


def _resolve_device(device: str) -> str:
    """device 参数解析：auto/cuda 不可用时回退 cpu。"""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _split_rgb_alpha(x):
    """拆出 RGB 与 alpha（若存在）。x: [B,H,W,C]"""
    if x.shape[-1] >= 4:
        return x[..., :3], x[..., 3:]
    return x, None


def _merge_rgb_alpha(rgb, alpha, ref):
    """按输入通道数还原输出，alpha 原样透传。"""
    if alpha is not None:
        return torch.cat([rgb, alpha], dim=-1)
    return rgb


def _gaussian_kernel(kernel_size, sigma, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def _gaussian_blur(x, kernel_size=9, sigma=3.0):
    """x: [B,H,W,C] -> 高斯模糊结果 [B,H,W,C]，纯 torch 实现。"""
    b, h, w, c = x.shape
    k = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    k = k[None, None, :, :].repeat(c, 1, 1, 1)  # [C,1,k,k]
    xt = x.permute(0, 3, 1, 2)                  # [B,C,H,W]
    pad = kernel_size // 2
    xt_padded = F.pad(xt, (pad, pad, pad, pad), mode="replicate")
    blurred = F.conv2d(xt_padded, k, groups=c)
    return blurred.permute(0, 2, 3, 1)          # [B,H,W,C]


class ColorMatchPlus:
    """精确颜色匹配：曝光(线性乘) -> 对比度 -> 色温/色调(通道白平衡偏移)。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "contrast": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "color_temp": ("FLOAT", {"default": 0.00, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tint": ("FLOAT", {"default": 0.00, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "JINGSI_Corrector"

    def apply(self, image, exposure=1.0, contrast=1.0, color_temp=0.0, tint=0.0, device="auto"):
        x, alpha = _split_rgb_alpha(image)
        dev = _resolve_device(device)
        x = x.to(dev)

        # 1) 曝光：线性乘
        out = x * exposure
        # 2) 对比度
        out = (out - 0.5) * contrast + 0.5
        # 3) 色温/色调：按通道做白平衡增益
        #    color_temp>0 暖(红增蓝减) <0 冷(红减蓝增)
        #    tint>0 品红(红蓝增绿减)  <0 绿(绿增红蓝减)
        r = out[..., 0] * (1.0 + 0.2 * color_temp + 0.15 * tint)
        g = out[..., 1] * (1.0 - 0.15 * tint)
        b = out[..., 2] * (1.0 - 0.2 * color_temp + 0.15 * tint)
        out = torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)

        out = out.to(image.device)
        return (_merge_rgb_alpha(out, alpha, image),)


class HighlightRestorePlus:
    """高光细节恢复：亮度掩码(平滑过渡)选取高光区，融合 original_image 的高频细节。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "strength": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "JINGSI_Corrector"

    def apply(self, image, original_image, threshold=0.5, strength=0.0, device="auto"):
        x, alpha = _split_rgb_alpha(image)
        orig, _ = _split_rgb_alpha(original_image)
        dev = _resolve_device(device)
        x = x.to(dev)
        orig = orig.to(dev)

        if strength <= 0.0 or threshold >= 1.0:
            out = x
        else:
            # 亮度掩码（线性平滑过渡）：luma>threshold 部分 -> [0,1]
            luma = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
            mask = (luma - threshold).clamp(0.0, 1.0).unsqueeze(-1)
            # 高频细节：原图减高斯模糊
            detail = orig - _gaussian_blur(orig)
            out = (x + mask * detail * strength).clamp(0.0, 1.0)

        out = out.to(image.device)
        return (_merge_rgb_alpha(out, alpha, image),)