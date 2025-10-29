import tkinter as tk
from tkinter import ttk
from typing import Any, List, Optional

def build(app):
    """Construct the Analysis view without a carousel:
    - Left: rendering area for the selected chart (analysis_display_frame)
    - Right: vertical list of chart names (analysis_chart_list)
    - Bottom: short Summary box (analysis_text, height=6)
    """
    container = ttk.Frame(app.analysis_tab)
    container.pack(fill="both", expand=True)

    # Header
    header = ttk.Frame(container)
    header.pack(fill="x", padx=8, pady=8)
    ttk.Label(header, text="Analysis", font=(None, 14, "bold")).pack(side="left", anchor="w")

    # Body: left (viewer) + right (chart list)
    body = ttk.Frame(container)
    body.pack(fill="both", expand=True)

    # Left: where the selected chart renders
    app.analysis_display_frame = ttk.Frame(body)
    app.analysis_display_frame.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=(0, 8))

    # Right: vertical list of chart buttons
    sidebar = ttk.Frame(body, width=260)
    sidebar.pack(side="right", fill="y", padx=(4, 8), pady=(0, 8))
    ttk.Label(sidebar, text="Charts").pack(anchor="w")

    # Frame to hold buttons with scrollbar
    button_frame = ttk.Frame(sidebar)
    button_frame.pack(fill="both", expand=True)
    
    # Canvas and scrollbar for button area
    canvas = tk.Canvas(button_frame)
    scrollbar = ttk.Scrollbar(button_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Store reference to scrollable frame for button creation
    app.analysis_chart_button_frame = scrollable_frame

    # Bottom: shorter Summary
    summary_frame = ttk.Frame(container)
    summary_frame.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Label(summary_frame, text="Summary").pack(anchor="w")
    app.analysis_text = tk.Text(summary_frame, height=6, wrap="word")
    app.analysis_text.insert("1.0", "Run a measurement to populate analysis.")
    app.analysis_text.configure(state="disabled")
    app.analysis_text.pack(fill="x", expand=False)
