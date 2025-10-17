import tkinter as tk
from tkinter import ttk
from typing import Any, List, Optional, Tuple, Dict


def build(app):
    """Construct the analysis tab layout that displays characterization charts."""

    container = ttk.Frame(app.analysis_tab)
    container.pack(fill="both", expand=True)

    header = ttk.Frame(container)
    header.pack(fill="x", padx=8, pady=8)

    ttk.Label(
        header,
        text="Spectrometer Characterization Results",
        font=("Segoe UI", 12, "bold"),
    ).pack(side="left")

    btns = ttk.Frame(header)
    btns.pack(side="right")
    app.export_plots_btn = ttk.Button(
        btns, text="Export Plots", command=app.export_analysis_plots, state="disabled"
    )
    app.export_plots_btn.pack(side="left", padx=4)
    app.open_folder_btn = ttk.Button(
        btns, text="Open Results Folder", command=app.open_results_folder, state="disabled"
    )
    app.open_folder_btn.pack(side="left", padx=4)

    app.analysis_status_var = tk.StringVar(value="Run measurements to generate characterization charts.")
    ttk.Label(
        container,
        textvariable=app.analysis_status_var,
        foreground="gray",
        wraplength=760,
        justify="left",
    ).pack(fill="x", padx=8)

    notebook_frame = ttk.Frame(container)
    notebook_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

    app.analysis_notebook = ttk.Notebook(notebook_frame)
    app.analysis_notebook.pack(fill="both", expand=True)
    app.analysis_canvases = []

    summary_frame = ttk.Frame(container)
    summary_frame.pack(fill="both", expand=False, padx=8, pady=8)
    ttk.Label(summary_frame, text="Summary").pack(anchor="w")
    app.analysis_text = tk.Text(summary_frame, height=12, wrap="word")
    app.analysis_text.insert("1.0", "No analysis has been generated yet. Run the measurement flow to populate this summary.")
    app.analysis_text.configure(state="disabled")
    app.analysis_text.pack(fill="both", expand=True)
