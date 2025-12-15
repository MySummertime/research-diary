# --- coding: utf-8 ---
# --- app/utils/visual_style.py ---
"""
[Visual Style Configuration] 统一视觉风格管理模块
集中管理所有绘图的颜色、marker、linewidth、fontsize 等，确保全项目配色一致、学术美观、色盲友好。
基于经典“黄昏渐变”配色系统，在浅色地图底图上高可见度。
"""

from typing import List, Dict


class ColorPalette:
    """
    “黄昏渐变”配色
    """

    # Hex 值
    DEFAULT_BLACK = "#000000"
    DEFAULT_GRAY = "#999999"
    DEFAULT_WHITE = "#FFFFFF"

    DARK_GRAY = "#999999"  # 深灰
    GRAY = "#CCCCCC"  # 灰
    LIGHT_GRAY = "#E4E4E4"  # 浅灰

    PURPLE_TWILIGHT = "#6A5ACD"  # 暮紫
    DUSK_PINK = "#D8A7B1"  # 黄昏粉

    DARK_BLUE = "#274753"  # 蓝灰
    OLIVE_GREEN = "#808000"  # 橄榄绿
    DEEP_TEAL = "#297270"  # 深青绿
    TEAL_GREEN = "#299d8f"  # 青绿
    LIGHT_GREEN = "#8ab07c"  # 嫩绿

    RED_ORANGE = "#e66d50"  # 红橙
    ORANGE = "#f3a361"  # 橙色
    SOFT_YELLOW = "#e7c66b"  # 明亮黄

    # ====================================================
    # === List[str] which can be achieved by id(int) ===
    # ====================================================

    #  --- Network tasks ---
    TASK_LOOP: List[str] = [
        PURPLE_TWILIGHT,
        DARK_BLUE,
        OLIVE_GREEN,
        ORANGE,
        DUSK_PINK,
    ]

    # -- Pareto Frontiers (loop) ---
    PARETO_LOOP: List[str] = [
        TEAL_GREEN,
        LIGHT_GREEN,
        SOFT_YELLOW,
        ORANGE,
        RED_ORANGE,
        DUSK_PINK,
    ]

    # --- Violin ---
    VIOLIN_LOOP: List[str] = [
        DARK_BLUE,
        DEEP_TEAL,
        SOFT_YELLOW,
        RED_ORANGE,
    ]

    # =====================================================
    # === List[Dict] which can be achieved by key(str) ===
    # =====================================================

    # --- Default colours ---
    DEFAULT_COLOR: List[Dict[str, str]] = [
        {"BLACK": DEFAULT_BLACK},
        {"GRAY": DEFAULT_GRAY},
        {"WHITE": DEFAULT_WHITE},
    ]

    # ---Network topology ---
    NETWORK_TOPOLOGY: List[Dict[str, str]] = [
        {"HUB": SOFT_YELLOW},
        {"NON_HUB": LIGHT_GREEN},
        {"EMERGENCY": RED_ORANGE},
        {"ROAD": DEFAULT_GRAY},
        {"RAILWAY": DEFAULT_BLACK},
    ]

    # --- Pareto Frontier (Single) ---
    PARETO: List[Dict[str, str]] = [
        {"SPECIAL_POINT": RED_ORANGE},
        {"PARETO_POINT_EDGE": DEEP_TEAL},
        {"PARETO_FRONT_LINE": TEAL_GREEN},
        {"RANK1_COLOR": DARK_GRAY},
        {"RANK2_COLOR": GRAY},
        {"RANK3_COLOR": LIGHT_GRAY},
    ]

    # -- Pareto Frontiers (by algorithm) ---
    PARETO_BY_ALGO: List[Dict[str, str]] = [
        {"EXACT": RED_ORANGE},
        {"PROPOSED": TEAL_GREEN},
        {"BASELINE1": SOFT_YELLOW},
        {"BASELINE2": DARK_BLUE},
    ]

    # --- Dual line chart ---
    DUAL_LINE: List[Dict[str, str]] = [
        {"COST_LINE": TEAL_GREEN},
        {"RISK_LINE": RED_ORANGE},
    ]

    # --- Stacked bar chart ---
    STACKED_BAR: List[Dict[str, str]] = [
        {"TRANSPORT_COLOR": DEEP_TEAL},
        {"TRANSSHIPMENT_COLOR": TEAL_GREEN},
        {"CARBON_COLOR": LIGHT_GREEN},
        {"RISK_COLOR": RED_ORANGE},
        {"TREND": DARK_BLUE},
    ]

    # --- Heat map ---
    HEATMAP: List[Dict[str, str]] = [
        {"COST_CMAP": "GnBu"},  # 成本热图（冷色表示低成本）
        {"RISK_CMAP": "YlOrBr"},  # 风险热图（暖色表示高风险）
    ]


def get_color_by_key(config_list: List, key: str) -> str:
    """
    从 [ { "KEY": color }, ... ] 结构中提取指定 key 的颜色
    """
    for item in config_list:
        if key in item:
            return item[key]
    raise KeyError(f"Color key '{key}' not found in config list.")
