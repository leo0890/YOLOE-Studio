"""
Custom Gradio theme for YOLOE service.

Единая точка стилизации: цвета, тени, типографика.
Не зависит от бизнес-логики — можно переиспользовать в любом проекте.
"""
from __future__ import annotations

import gradio as gr


# --- палитра (единый источник правды для UI) ---
class Palette:
    PRIMARY = "#6366f1"          # indigo-500
    PRIMARY_DARK = "#4f46e5"     # indigo-600
    PRIMARY_LIGHT = "#a5b4fc"    # indigo-300

    SUCCESS = "#10b981"          # emerald-500
    WARNING = "#f59e0b"          # amber-500
    DANGER = "#ef4444"           # red-500
    INFO = "#3b82f6"             # blue-500

    BG = "#f8fafc"               # slate-50
    SURFACE = "#ffffff"
    BORDER = "#e2e8f0"           # slate-200

    TEXT = "#0f172a"             # slate-900
    TEXT_MUTED = "#64748b"       # slate-500


def build_theme() -> gr.themes.Base:
    """Собирает кастомную тему Gradio."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill=Palette.BG,
        body_text_color=Palette.TEXT,

        block_background_fill=Palette.SURFACE,
        block_border_color=Palette.BORDER,
        block_border_width="1px",
        block_radius="12px",
        block_shadow="0 1px 3px rgba(0,0,0,0.05)",

        button_primary_background_fill=Palette.PRIMARY,
        button_primary_background_fill_hover=Palette.PRIMARY_DARK,
        button_primary_text_color="#ffffff",
        button_primary_border_color=Palette.PRIMARY,

        button_secondary_background_fill=Palette.SURFACE,
        button_secondary_background_fill_hover="#f1f5f9",
        button_secondary_text_color=Palette.TEXT,
        button_secondary_border_color=Palette.BORDER,

        input_background_fill=Palette.SURFACE,
        input_border_color=Palette.BORDER,
        input_border_width="1px",
        input_radius="8px",

        slider_color=Palette.PRIMARY,
        checkbox_background_color_selected=Palette.PRIMARY,

        panel_background_fill=Palette.SURFACE,
        panel_border_color=Palette.BORDER,
    )