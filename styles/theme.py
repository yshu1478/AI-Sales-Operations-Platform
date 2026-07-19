from __future__ import annotations

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


COLORS = {
    "primary": "#3370FF",
    "primary_hover": "#2859D9",
    "success": "#22A06B",
    "warning": "#F59E0B",
    "danger": "#E5484D",
    "background": "#F7F8FA",
    "surface": "#FFFFFF",
    "text": "#1F2329",
    "muted": "#646A73",
    "border": "#E5E6EB",
    "grid": "#EEF0F3",
}

PLOTLY_COLORWAY = ["#3370FF", "#14C9C9", "#7C5CFC", "#22A06B", "#F59E0B", "#E5484D"]
PLOTLY_SEQUENTIAL = [[0.0, "#E8F0FF"], [0.45, "#85A9FF"], [1.0, "#3370FF"]]
FONT_FAMILY = 'Inter, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif'


def load_theme() -> None:
    """从单一 CSS 文件加载全站主题。"""
    css_path = Path(__file__).with_name("style.css")
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="saas-brand">
            <div class="saas-brand-mark">S</div>
            <div>
                <div class="saas-brand-title">AI Sales Ops</div>
                <div class="saas-brand-subtitle">销售运营平台</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, breadcrumb: str, description: str | None = None) -> None:
    description_html = f'<p class="saas-page-description">{escape(description)}</p>' if description else ""
    st.markdown(
        f"""
        <header class="saas-page-header">
            <div class="saas-breadcrumb">{escape(breadcrumb)}</div>
            <h1>{escape(title)}</h1>
            {description_html}
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_timeline_item(item_date: object, item_type: str, content: str) -> None:
    formatted_date = item_date.strftime("%Y-%m-%d") if hasattr(item_date, "strftime") else str(item_date)
    st.markdown(
        f"""
        <div class="saas-timeline-item">
            <span class="saas-timeline-dot"></span>
            <div class="saas-timeline-content">
                <div class="saas-timeline-meta">{escape(formatted_date)} · {escape(str(item_type))}</div>
                <div class="saas-timeline-text">{escape(str(content))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(figure: go.Figure) -> go.Figure:
    """为所有 Plotly 图表应用一致的企业后台视觉规范。"""
    figure.update_layout(
        colorway=PLOTLY_COLORWAY,
        font={"family": FONT_FAMILY, "size": 13, "color": COLORS["muted"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["surface"],
        margin={"l": 24, "r": 24, "t": 28, "b": 20},
        hoverlabel={
            "bgcolor": COLORS["text"],
            "bordercolor": COLORS["text"],
            "font": {"family": FONT_FAMILY, "size": 13, "color": COLORS["surface"]},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": COLORS["muted"]},
        },
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"]},
    )
    return figure
