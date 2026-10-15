from .nodes import ColorMatchPlus, HighlightRestorePlus

NODE_CLASS_MAPPINGS = {
    "ColorMatchPlus": ColorMatchPlus,
    "HighlightRestorePlus": HighlightRestorePlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorMatchPlus": "精确颜色匹配",
    "HighlightRestorePlus": "高光细节恢复",
}

__version__ = "1.0.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]