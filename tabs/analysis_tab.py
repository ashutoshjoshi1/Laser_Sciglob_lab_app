import tkinter as tk
from tkinter import ttk
from typing import Any, List, Optional, Tuple, Dict

# Optional Pillow support for JPG/resize; falls back to Tk.PhotoImage if unavailable
try:
    from PIL import Image, ImageTk  # type: ignore
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


class ImageCarousel(ttk.Frame):
    """A simple image carousel widget with Prev/Next and optional autoplay.

    Usage:
        carousel = ImageCarousel(parent, interval_ms=3000, autoplay=False)
        carousel.pack(fill="both", expand=True)
        carousel.set_images(["chart1.png", "chart2.png"])  # file paths OR preloaded PhotoImage

        # Later, from your analysis pipeline:
        app.analysis_carousel.set_images(new_paths_or_images)
    """

    def __init__(
        self,
        master: tk.Misc,
        images: Optional[List[Any]] = None,
        interval_ms: int = 3000,
        autoplay: bool = False,
        bg: str = "#1e1e1e",
    ) -> None:
        super().__init__(master)
        self.configure(style="Carousel.TFrame")

        self._raw_inputs: List[Any] = []  # str paths or PhotoImage/PIL Image
        self._pil_images: List["Image.Image"] = []  # only if PIL available
        self._tk_images: List[tk.PhotoImage] = []  # cached, resized to fit label
        self._index: int = 0
        self._job: Optional[str] = None
        self._interval = interval_ms
        self._autoplay = autoplay
        self._bg = bg

        # Styles
        style = ttk.Style(self)
        try:
            style.configure("Carousel.TFrame", background=self._bg)
            style.configure("Carousel.TLabel", background=self._bg, foreground="#ddd")
            style.configure("Carousel.TButton", padding=6)
        except Exception:
            pass

        # Layout
        self._top = ttk.Frame(self, style="Carousel.TFrame")
        self._top.pack(fill="x", padx=8, pady=(8, 0))

        self._title = ttk.Label(self._top, text="Image Carousel", style="Carousel.TLabel", font=(None, 12, "bold"))
        self._title.pack(side="left")

        self._controls = ttk.Frame(self, style="Carousel.TFrame")
        self._controls.pack(fill="x", padx=8, pady=(4, 0))

        self._prev_btn = ttk.Button(self._controls, text="◀ Prev", command=self.prev, style="Carousel.TButton")
        self._prev_btn.pack(side="left")

        self._play_btn = ttk.Button(self._controls, text="▶ Play" if not self._autoplay else "⏸ Pause", command=self.toggle_autoplay, style="Carousel.TButton")
        self._play_btn.pack(side="left", padx=(6, 0))

        self._next_btn = ttk.Button(self._controls, text="Next ▶", command=self.next, style="Carousel.TButton")
        self._next_btn.pack(side="left", padx=(6, 0))

        self._counter = ttk.Label(self._controls, text="0 / 0", style="Carousel.TLabel")
        self._counter.pack(side="right")

        self._viewport = tk.Label(self, bg=self._bg, bd=0, highlightthickness=0, anchor="center")
        self._viewport.pack(fill="both", expand=True, padx=8, pady=8)

        # Resize handling
        self.bind("<Configure>", self._on_resize)

        if images:
            self.set_images(images)

        if self._autoplay:
            self._schedule_next()

    # ---------------------------- Public API ---------------------------- #

    def set_images(self, images: List[Any]) -> None:
        """Accept file paths or already loaded images (PhotoImage or PIL.Image)."""
        self._cancel_autoplay()
        self._raw_inputs = images or []
        self._index = 0
        self._load_internal_images()
        self._render_current()
        if self._autoplay:
            self._schedule_next()

    # --------------------------- Internals ----------------------------- #

    def _load_internal_images(self) -> None:
        self._pil_images.clear()
        self._tk_images.clear()

        for item in self._raw_inputs:
            if isinstance(item, tk.PhotoImage):
                self._tk_images.append(item)
                if _PIL_AVAILABLE:
                    try:
                        # Best-effort conversion from PhotoImage -> PIL via bytes (PNG) if possible
                        # Not guaranteed; we rely on Tk-only rendering in that case.
                        pass
                    except Exception:
                        pass
            elif _PIL_AVAILABLE and hasattr(item, "size") and hasattr(item, "mode"):
                # Looks like a PIL.Image.Image
                self._pil_images.append(item)
                self._tk_images.append(None)  # placeholder for resized cache
            elif isinstance(item, str):
                # file path
                if _PIL_AVAILABLE:
                    try:
                        img = Image.open(item)
                        self._pil_images.append(img)
                        self._tk_images.append(None)
                    except Exception:
                        # Fall back to Tk PhotoImage (supports GIF/PNG)
                        try:
                            self._tk_images.append(tk.PhotoImage(file=item))
                        except Exception:
                            # skip bad file
                            continue
                else:
                    try:
                        self._tk_images.append(tk.PhotoImage(file=item))
                    except Exception:
                        continue

        # If no images successfully loaded, ensure consistent state
        if not self._pil_images and not self._tk_images:
            self._index = 0
            self._counter.configure(text="0 / 0")
            self._viewport.configure(text="No images to display", fg="#999")
        else:
            self._counter.configure(text=f"1 / {self._count}")

    @property
    def _count(self) -> int:
        return max(len(self._tk_images), len(self._pil_images)) or 0

    def _render_current(self) -> None:
        if self._count == 0:
            self._viewport.configure(text="No images to display", image="", fg="#999")
            self._counter.configure(text="0 / 0")
            return

        w = max(self._viewport.winfo_width(), 10)
        h = max(self._viewport.winfo_height(), 10)

        img = self._get_tk_image(self._index, (w, h))
        if img is not None:
            self._viewport.configure(image=img, text="")
            self._viewport.image = img  # prevent GC
        else:
            self._viewport.configure(text="(Image unsupported)", image="", fg="#999")

        self._counter.configure(text=f"{self._index + 1} / {self._count}")

    def _get_tk_image(self, idx: int, size: Tuple[int, int]) -> Optional[tk.PhotoImage]:
        # Try cached/resized first
        if _PIL_AVAILABLE and idx < len(self._pil_images) and self._pil_images[idx] is not None:
            pil = self._pil_images[idx]
            try:
                target = self._fit_image(pil, size)
                return ImageTk.PhotoImage(target)
            except Exception:
                return None

        if idx < len(self._tk_images):
            return self._tk_images[idx]
        return None

    def _fit_image(self, pil_img: "Image.Image", size: Tuple[int, int]) -> "Image.Image":
        vw, vh = size
        vw = max(vw - 16, 1)  # padding allowance
        vh = max(vh - 16, 1)
        iw, ih = pil_img.size
        if iw == 0 or ih == 0:
            return pil_img
        scale = min(vw / iw, vh / ih)
        nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
        try:
            return pil_img.resize((nw, nh), Image.LANCZOS)
        except Exception:
            return pil_img.resize((nw, nh))

    def _on_resize(self, _evt: tk.Event) -> None:
        # Re-render current image to fit new size
        self.after_idle(self._render_current)

    def _schedule_next(self) -> None:
        self._job = self.after(self._interval, self._tick)

    def _cancel_autoplay(self) -> None:
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _tick(self) -> None:
        self.next()
        if self._autoplay:
            self._schedule_next()

    # --------------------------- Controls ------------------------------ #

    def next(self) -> None:
        if self._count == 0:
            return
        self._index = (self._index + 1) % self._count
        self._render_current()

    def prev(self) -> None:
        if self._count == 0:
            return
        self._index = (self._index - 1) % self._count
        self._render_current()

    def toggle_autoplay(self) -> None:
        self._autoplay = not self._autoplay
        self._play_btn.configure(text="⏸ Pause" if self._autoplay else "▶ Play")
        if self._autoplay:
            self._schedule_next()
        else:
            self._cancel_autoplay()


def build(app):
    """Construct the analysis tab layout that displays characterization charts and an image carousel."""

    container = ttk.Frame(app.analysis_tab)
    container.pack(fill="both", expand=True)

    header = ttk.Frame(container)
    header.pack(fill="x", padx=8, pady=8)

    ttk.Label(
        header,
        text="Analysis",
        font=(None, 14, "bold"),
    ).pack(side="left", anchor="w")

    # --------------------- Carousel Section --------------------- #
    carousel_holder = ttk.Frame(container)
    carousel_holder.pack(fill="both", expand=False, padx=8, pady=(0, 8))

    app.analysis_carousel = ImageCarousel(carousel_holder, interval_ms=3000, autoplay=False)
    app.analysis_carousel.pack(fill="both", expand=True)

    # --------------------- Notebook with Charts --------------------- #
    notebook_frame = ttk.Frame(container)
    notebook_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

    app.analysis_notebook = ttk.Notebook(notebook_frame)
    app.analysis_notebook.pack(fill="both", expand=True)
    app.analysis_canvases = []

    # --------------------- Summary (existing) --------------------- #
    summary_frame = ttk.Frame(container)
    summary_frame.pack(fill="both", expand=False, padx=8, pady=8)
    ttk.Label(summary_frame, text="Summary").pack(anchor="w")
    app.analysis_text = tk.Text(summary_frame, height=12, wrap="word")
    app.analysis_text.insert("1.0", "No analysis has been generated yet. Run the measurement flow to populate this summary.")
    app.analysis_text.configure(state="disabled")
    app.analysis_text.pack(fill="both", expand=True)
