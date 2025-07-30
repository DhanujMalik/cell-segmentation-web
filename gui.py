import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import logging
from datetime import datetime
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import gabor, gaussian, laplace, sobel
from sklearn.ensemble import RandomForestClassifier
import re
import shutil
from glob import glob
import colorsys
from collections import OrderedDict
import threading
import queue
import time


class EnhancedScrollFrame(ttk.Frame):
    """Custom scrollable frame widget"""

    def _init_(self, parent, *args, **kwargs):
        ttk.Frame._init_(self, parent, *args, **kwargs)

        # Create canvas and scrollbars
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hscroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vscroll.set, xscrollcommand=self.hscroll.set)

        # Layout
        self.vscroll.pack(side="right", fill="y")
        self.hscroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Create interior frame
        self.interior = ttk.Frame(self.canvas)
        self.interior_id = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")

        # Bind events
        self.interior.bind('<Configure>', self._configure_interior)
        self.canvas.bind('<Configure>', self._configure_canvas)
        self.canvas.bind('<Enter>', self._bind_to_mousewheel)
        self.canvas.bind('<Leave>', self._unbind_from_mousewheel)

    def _configure_interior(self, event):
        """Update scrollregion when interior size changes"""
        size = (self.interior.winfo_reqwidth(), self.interior.winfo_reqheight())
        self.canvas.config(scrollregion="0 0 %s %s" % size)
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            self.canvas.config(width=self.interior.winfo_reqwidth())

    def _configure_canvas(self, event):
        """Update interior width when canvas size changes"""
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            self.canvas.itemconfigure(self.interior_id, width=self.canvas.winfo_width())

    def _bind_to_mousewheel(self, event):
        """Bind mousewheel to scrolling"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_from_mousewheel(self, event):
        """Unbind mousewheel when leaving canvas"""
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class AdvancedSegmentationApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Advanced Interactive Segmentation Tool")
        self.root.geometry("1400x900")
        self.root.state('zoomed')

        # Configure logging
        self.setup_logging()

        # Initialize state variables
        self._initialize_state_variables()

        # Initialize UI
        self._initialize_ui()

        # Create directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.frames_folder, exist_ok=True)

        self.update_ui_state()

    def _initialize_state_variables(self):
        """Initialize all state variables"""
        # Path variables
        self.input_path = ""
        self.output_folder = "output_segments"
        self.frames_folder = "frames"
        self.training_image_path = None

        # Image variables
        self.current_image = None
        self.processed_image = None
        self.original_image = None
        self.reference_image = None
        self.img_tk = None
        self.img_width = 0
        self.img_height = 0

        # Labeling variables
        self.label_mask = None
        self.current_label = 1
        self.label_colors = OrderedDict([
            (1, (255, 0, 0)),  # Red - Cell
            (2, (0, 0, 0)),  # Black - Background
            (3, (0, 255, 0)),  # Green - Nucleus
            (4, (0, 0, 255))  # Blue - Membrane
        ])
        self.label_names = {
            1: "Cell",
            2: "Background",
            3: "Nucleus",
            4: "Membrane"
        }
        self.next_label_id = 5

        # Video variables
        self.video_frames = []
        self.frame_interval = 1
        self.video_cap = None
        self.video_fps = 0
        self.video_duration = 0
        self.extraction_mode = None
        self.selected_frame_index = 0
        self.max_frames = 100

        # Processing variables
        self.classifier = None
        self.feature_params = {}
        self.zoom_level = 1.0
        self.crop_coords = None
        self.crop_mode = False
        self.crop_rect_id = None
        self.crop_points = []
        self.current_step = 0
        self.current_tool = "brush"
        self.last_x = None
        self.last_y = None

        # Batch processing
        self.batch_queue = queue.Queue()
        self.batch_running = False
        self.batch_thread = None

        # Tkinter variables
        self._initialize_tk_vars()

    def _initialize_tk_vars(self):
        """Initialize all Tkinter variables"""
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.sharpness_var = tk.DoubleVar(value=1.0)
        self.denoise_var = tk.IntVar(value=0)
        self.live_update_var = tk.IntVar(value=0)
        self.suggest_features_var = tk.IntVar(value=0)
        self.binary_output_var = tk.IntVar(value=1)
        self.brush_size = tk.IntVar(value=5)
        self.crop_var = tk.IntVar(value=0)
        self.train_crop_var = tk.IntVar(value=0)
        self.output_format = tk.StringVar(value="PNG")
        self.overwrite_var = tk.IntVar(value=0)
        self.save_probabilities = tk.IntVar(value=0)
        self.save_features = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        self.input_type = tk.StringVar(value="image")

    def _initialize_ui(self):
        """Initialize the main UI components"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel with scrollbar
        left_panel = EnhancedScrollFrame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Right panel - display area
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create sections in left panel
        self._create_input_section(left_panel.interior)
        self._create_preprocess_section(left_panel.interior)
        self._create_feature_section(left_panel.interior)
        self._create_labeling_section(left_panel.interior)
        self._create_segmentation_section(left_panel.interior)
        self._create_batch_section(left_panel.interior)

        # Display area
        self._create_display_area(right_panel)

        # Status bar
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

    def _create_display_area(self, parent):
        """Create the image display area"""
        display_frame = ttk.LabelFrame(parent, text="Display", padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas with scrollbars
        self.canvas_frame = ttk.Frame(display_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.hscroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.vscroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(self.canvas_frame, bg='white',
                                xscrollcommand=self.hscroll.set,
                                yscrollcommand=self.vscroll.set,
                                highlightthickness=0)
        self.hscroll.config(command=self.canvas.xview)
        self.vscroll.config(command=self.canvas.yview)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.hscroll.grid(row=1, column=0, sticky="ew")

        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        # Zoom controls
        zoom_frame = ttk.Frame(display_frame)
        zoom_frame.pack(fill=tk.X, pady=5)
        ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.adjust_zoom(1.25)).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.adjust_zoom(0.8)).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Fit to Window", command=self.fit_to_window).pack(side=tk.LEFT)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint_label)
        self.canvas.bind("<Button-1>", self.paint_label)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_coords)

    def _create_input_section(self, parent):
        """Create input selection section"""
        self.input_frame = ttk.LabelFrame(parent, text="1. Input Selection", padding=10)
        self.input_frame.pack(fill=tk.X, pady=5)

        # Input type selection
        ttk.Radiobutton(self.input_frame, text="Single Image", variable=self.input_type, value="image").pack(
            anchor=tk.W)
        ttk.Radiobutton(self.input_frame, text="Image Folder", variable=self.input_type, value="folder").pack(
            anchor=tk.W)
        ttk.Radiobutton(self.input_frame, text="Video File", variable=self.input_type, value="video").pack(anchor=tk.W)

        # Action buttons
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Browse", command=self.load_input).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Reset", command=self.reset_input).pack(side=tk.RIGHT)

        self.input_label = ttk.Label(self.input_frame, text="No file selected")
        self.input_label.pack()

        # Image selection for folders
        self.image_selection_frame = ttk.LabelFrame(self.input_frame, text="Image Selection", padding=10)
        ttk.Label(self.image_selection_frame, text="Select training image:").pack(anchor=tk.W)
        self.image_selector = ttk.Combobox(self.image_selection_frame, state='readonly')
        self.image_selector.pack(fill=tk.X)
        ttk.Button(self.image_selection_frame, text="Select", command=self.load_selected_folder_image).pack(pady=5)
        self.image_selection_frame.pack_forget()

        # Colony preview button
        self.colony_preview_btn = ttk.Button(self.input_frame, text="Preview Colony Growth",
                                             command=self.show_colony_growth_preview, state=tk.DISABLED)
        self.colony_preview_btn.pack(pady=5)

    def _create_preprocess_section(self, parent):
        """Create preprocessing section"""
        self.preprocess_frame = ttk.LabelFrame(parent, text="2. Preprocessing", padding=10)
        self.preprocess_frame.pack(fill=tk.X, pady=5)

        # Enhancement controls
        enhance_frame = ttk.LabelFrame(self.preprocess_frame, text="Image Enhancement", padding=5)
        enhance_frame.pack(fill=tk.X, pady=5)

        controls = [
            ("Brightness:", self.brightness_var),
            ("Contrast:", self.contrast_var),
            ("Sharpness:", self.sharpness_var)
        ]

        for text, var in controls:
            ttk.Label(enhance_frame, text=text).pack(anchor=tk.W)
            ttk.Scale(enhance_frame, from_=0.1, to=2.0, variable=var,
                      command=lambda e: self.adjust_image_quality()).pack(fill=tk.X)

        ttk.Checkbutton(enhance_frame, text="Denoise", variable=self.denoise_var,
                        command=self.adjust_image_quality).pack(anchor=tk.W)

        ttk.Button(enhance_frame, text="Reset Enhancements", command=self.reset_enhancements).pack(pady=5)

        # Crop controls
        crop_frame = ttk.Frame(self.preprocess_frame)
        crop_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(crop_frame, text="Enable Cropping", variable=self.crop_var,
                        command=self.toggle_crop).pack(side=tk.LEFT)
        self.crop_btn = ttk.Button(crop_frame, text="Select Area", command=self.start_crop_selection, state=tk.DISABLED)
        self.crop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(crop_frame, text="Reset Crop", command=self.reset_crop).pack(side=tk.RIGHT)

        ttk.Checkbutton(self.preprocess_frame, text="Apply same crop to all images",
                        variable=self.train_crop_var).pack(anchor=tk.W)

    def _create_feature_section(self, parent):
        """Create feature selection section"""
        self.feature_frame = ttk.LabelFrame(parent, text="3. Feature Selection", padding=10)
        self.feature_frame.pack(fill=tk.X, pady=5)

        # Sigma parameters
        sigma_frame = ttk.Frame(self.feature_frame)
        sigma_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sigma_frame, text="Sigma:").pack(side=tk.LEFT)
        self.sigma_vars = []
        default_sigmas = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]

        for sigma in default_sigmas:
            var = tk.DoubleVar(value=sigma)
            self.sigma_vars.append(var)
            ttk.Entry(sigma_frame, textvariable=var, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Button(sigma_frame, text="Add", command=self.add_sigma).pack(side=tk.LEFT, padx=5)

        # Feature checkboxes
        features = [
            "Gaussian Smoothing",
            "Edge",
            "Laplacian of Gaussian",
            "Gaussian Gradient Magnitude",
            "Difference of Gaussians",
            "Texture",
            "Structure Tensor Eigenvalues",
            "Hessian of Gaussian Eigenvalue"
        ]

        for feature in features:
            var = tk.IntVar(value=0)
            self.feature_params[feature] = {"var": var}
            ttk.Checkbutton(self.feature_frame, text=feature, variable=var).pack(anchor=tk.W)

    def _create_labeling_section(self, parent):
        """Create labeling section"""
        self.label_frame = ttk.LabelFrame(parent, text="4. Training", padding=10)
        self.label_frame.pack(fill=tk.X, pady=5)

        # Current label display
        current_label_frame = ttk.Frame(self.label_frame)
        current_label_frame.pack(fill=tk.X, pady=5)

        ttk.Label(current_label_frame, text="Current Label:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
        self.current_label_text = ttk.Label(current_label_frame, text="Cell (1)")
        self.current_label_text.pack(side=tk.LEFT, padx=5)
        ttk.Button(current_label_frame, text="+ Add Label", command=self.add_label).pack(side=tk.RIGHT)

        ttk.Separator(self.label_frame).pack(fill=tk.X, pady=5)

        # Tools section
        tools_frame = ttk.LabelFrame(self.label_frame, text="Labeling Tools", padding=5)
        tools_frame.pack(fill=tk.X, pady=5)

        ttk.Label(tools_frame, text="Brush Size:").pack(anchor=tk.W)
        ttk.Scale(tools_frame, from_=1, to=50, variable=self.brush_size, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Tool buttons
        btn_frame = ttk.Frame(tools_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Label(btn_frame, text="•").pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Brush", command=lambda: self.set_tool("brush")).pack(side=tk.LEFT, padx=2)
        ttk.Label(btn_frame, text="•").pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Erase", command=lambda: self.set_tool("erase")).pack(side=tk.LEFT, padx=2)
        ttk.Label(btn_frame, text="•").pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Reset", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)

        ttk.Separator(self.label_frame).pack(fill=tk.X, pady=5)

        # Action buttons
        action_frame = ttk.Frame(self.label_frame)
        action_frame.pack(fill=tk.X, pady=5)

        clear_frame = ttk.Frame(action_frame)
        clear_frame.pack(fill=tk.X, pady=2)
        ttk.Label(clear_frame, text="•").pack(side=tk.LEFT)
        ttk.Button(clear_frame, text="Clear Labels", command=self.clear_labels).pack(side=tk.LEFT, padx=2)

        train_frame = ttk.Frame(action_frame)
        train_frame.pack(fill=tk.X, pady=2)
        ttk.Label(train_frame, text="•").pack(side=tk.LEFT)
        ttk.Button(train_frame, text="Train Classifier", command=self.train_classifier).pack(side=tk.LEFT, padx=2)

        # Checkboxes
        checkbox_frame = ttk.Frame(self.label_frame)
        checkbox_frame.pack(fill=tk.X, pady=5)

        suggest_frame = ttk.Frame(checkbox_frame)
        suggest_frame.pack(fill=tk.X, pady=2)
        ttk.Label(suggest_frame, text="•").pack(side=tk.LEFT)
        ttk.Checkbutton(suggest_frame, text="Suggest Features", variable=self.suggest_features_var,
                        command=self.toggle_feature_suggestion).pack(side=tk.LEFT, padx=2)

        live_frame = ttk.Frame(checkbox_frame)
        live_frame.pack(fill=tk.X, pady=2)
        ttk.Label(live_frame, text="•").pack(side=tk.LEFT)
        ttk.Checkbutton(live_frame, text="Live Update", variable=self.live_update_var,
                        command=self.toggle_live_update).pack(side=tk.LEFT, padx=2)

        # Label preview
        self.label_preview = tk.Canvas(self.label_frame, width=20, height=20, bg='white', bd=1, relief='solid')
        self.label_preview.pack(pady=5)
        self.update_label_preview()

    def _create_segmentation_section(self, parent):
        """Create segmentation section"""
        self.segment_frame = ttk.LabelFrame(parent, text="5. Segmentation", padding=10)
        self.segment_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(self.segment_frame, text="Binary Output (White cells/Black background)",
                        variable=self.binary_output_var).pack(anchor=tk.W)

        btn_frame = ttk.Frame(self.segment_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Preview", command=self.preview_segmentation).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Process Current", command=self.process_current_image).pack(side=tk.RIGHT)

        self.progress = ttk.Progressbar(self.segment_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

    def _create_batch_section(self, parent):
        """Create batch processing section"""
        self.batch_frame = ttk.LabelFrame(parent, text="6. Batch Processing", padding=10)
        self.batch_frame.pack(fill=tk.X, pady=5)

        # Output format
        format_frame = ttk.Frame(self.batch_frame)
        format_frame.pack(fill=tk.X, pady=2)
        ttk.Label(format_frame, text="Output Format:").pack(side=tk.LEFT)
        ttk.Radiobutton(format_frame, text="PNG", variable=self.output_format, value="PNG").pack(side=tk.LEFT)
        ttk.Radiobutton(format_frame, text="TIFF", variable=self.output_format, value="TIFF").pack(side=tk.LEFT)
        ttk.Radiobutton(format_frame, text="JPG", variable=self.output_format, value="JPG").pack(side=tk.LEFT)

        # Options
        ttk.Checkbutton(self.batch_frame, text="Overwrite existing files",
                        variable=self.overwrite_var).pack(anchor=tk.W)
        ttk.Checkbutton(self.batch_frame, text="Save probability maps",
                        variable=self.save_probabilities).pack(anchor=tk.W)
        ttk.Checkbutton(self.batch_frame, text="Save feature stacks",
                        variable=self.save_features).pack(anchor=tk.W)

        # Buttons
        btn_frame = ttk.Frame(self.batch_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Select Output Folder",
                   command=self.select_output_folder).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Run Batch Processing",
                   command=self.start_batch_processing).pack(side=tk.RIGHT)

        # Progress
        self.batch_progress = ttk.Progressbar(self.batch_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=5)

        self.batch_status = ttk.Label(self.batch_frame, text="Ready for batch processing")
        self.batch_status.pack()

        self.stop_button = ttk.Button(self.batch_frame, text="Stop",
                                      command=self.stop_batch_processing, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

    def setup_logging(self):
        """Configure logging system"""
        log_folder = "logs"
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )

    def clear_output_folders(self):
        """Clear all output folders"""
        for folder in [self.output_folder, self.frames_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

    def load_input(self):
        """Load input file or folder based on selection"""
        try:
            input_type = self.input_type.get()

            if input_type == "image":
                self.input_path = filedialog.askopenfilename(
                    title="Select Image File",
                    filetypes=[("Image Files", ".png *.jpg *.jpeg *.bmp *.tiff"), ("All Files", ".*")]
                )
                if self.input_path:
                    self.input_label.config(text=os.path.basename(self.input_path))
                    self.load_image()

            elif input_type == "folder":
                self.input_path = filedialog.askdirectory(title="Select Image Folder")
                if self.input_path:
                    self.input_label.config(text=os.path.basename(self.input_path))
                    self.load_image_folder()

            elif input_type == "video":
                self.input_path = filedialog.askopenfilename(
                    title="Select Video File",
                    filetypes=[("Video Files", ".mp4 *.avi *.mov *.mkv"), ("All Files", ".*")]
                )
                if self.input_path:
                    self.input_label.config(text=os.path.basename(self.input_path))
                    self.video_cap = cv2.VideoCapture(self.input_path)
                    if not self.video_cap.isOpened():
                        raise ValueError("Failed to open video file")

                    self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.video_duration = frame_count / self.video_fps
                    self.video_cap.release()

                    self.show_video_settings_dialog()

        except Exception as e:
            logging.error(f"Input loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load input: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def show_video_settings_dialog(self):
        """Show dialog to configure video extraction settings"""
        if not self.input_path or not self.input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return

        # Create settings window
        self.video_settings_window = tk.Toplevel(self.root)
        self.video_settings_window.title("Video Extraction Settings")
        self.video_settings_window.geometry("400x400")
        self.video_settings_window.grab_set()

        # Main container with scrollbar
        main_container = EnhancedScrollFrame(self.video_settings_window)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Extraction mode selection
        self.extraction_mode = tk.StringVar(value="frame")
        mode_frame = ttk.LabelFrame(main_container.interior, text="Extraction Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(mode_frame, text="By Frames", variable=self.extraction_mode,
                        value="frame", command=self.update_extraction_ui).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="By Time (seconds)", variable=self.extraction_mode,
                        value="time", command=self.update_extraction_ui).pack(anchor=tk.W)

        # Frame-based settings
        self.frame_settings = ttk.LabelFrame(main_container.interior, text="Frame Settings", padding=10)

        ttk.Label(self.frame_settings, text="Frame interval:").pack(anchor=tk.W)
        self.interval_entry = ttk.Entry(self.frame_settings)
        self.interval_entry.pack(fill=tk.X)
        self.interval_entry.insert(0, "1")

        ttk.Label(self.frame_settings, text="Max frames to extract:").pack(anchor=tk.W)
        self.max_frames_entry = ttk.Entry(self.frame_settings)
        self.max_frames_entry.pack(fill=tk.X)
        self.max_frames_entry.insert(0, "50")

        self.frame_settings.pack(fill=tk.X, pady=5)

        # Time-based settings (initially hidden)
        self.time_settings = ttk.LabelFrame(main_container.interior, text="Time Settings", padding=10)

        ttk.Label(self.time_settings, text="Time interval (seconds):").pack(anchor=tk.W)
        self.time_interval_entry = ttk.Entry(self.time_settings)
        self.time_interval_entry.pack(fill=tk.X)
        self.time_interval_entry.insert(0, "1")

        ttk.Label(self.time_settings, text="Max duration (seconds):").pack(anchor=tk.W)
        self.max_duration_entry = ttk.Entry(self.time_settings)
        self.max_duration_entry.pack(fill=tk.X)
        self.max_duration_entry.insert(0, "60")

        self.time_settings.pack_forget()

        # Video info
        info_frame = ttk.Frame(main_container.interior)
        info_frame.pack(fill=tk.X, pady=5)

        if self.video_fps > 0:
            ttk.Label(info_frame, text=f"Video FPS: {self.video_fps:.2f}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Duration: {self.video_duration:.2f} sec").pack(anchor=tk.W)

        # Action buttons
        btn_frame = ttk.Frame(main_container.interior)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Cancel", command=self.video_settings_window.destroy).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Extract Frames", command=self.process_video_settings).pack(side=tk.RIGHT)

    def update_extraction_ui(self):
        """Update UI based on extraction mode selection"""
        if self.extraction_mode.get() == "frame":
            self.frame_settings.pack(fill=tk.X, pady=5)
            self.time_settings.pack_forget()
        else:
            self.frame_settings.pack_forget()
            self.time_settings.pack(fill=tk.X, pady=5)

    def process_video_settings(self):
        """Process the video with selected settings"""
        try:
            if self.extraction_mode.get() == "frame":
                frame_interval = max(1, int(self.interval_entry.get()))
                max_frames = max(1, int(self.max_frames_entry.get()))
            else:
                time_interval = max(0.1, float(self.time_interval_entry.get()))
                max_duration = max(1, float(self.max_duration_entry.get()))
                frame_interval = int(time_interval * self.video_fps)
                max_frames = int(max_duration * self.video_fps / frame_interval)

            self.load_video_frames(frame_interval, max_frames)
            self.video_settings_window.destroy()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def load_video_frames(self, frame_interval=1, max_frames=50):
        """Load frames from video with specified settings"""
        try:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video")

            self.video_frames = []
            frame_count = 0
            frames_loaded = 0

            while cap.isOpened() and frames_loaded < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_count / self.video_fps
                    self.video_frames.append((frame_count, frame, f"frame_{frame_count:04d}.png", timestamp))
                    frames_loaded += 1

                frame_count += 1

            cap.release()

            if not self.video_frames:
                raise ValueError("No frames loaded from video")

            self.show_frame_selector()
            self.colony_preview_btn.config(state=tk.NORMAL)

        except Exception as e:
            logging.error(f"Video loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")

    def show_frame_selector(self):
        """Show dialog to select reference frame"""
        if not self.video_frames:
            messagebox.showwarning("Warning", "No video frames loaded")
            return

        selector = tk.Toplevel(self.root)
        selector.title("Select Reference Frame")
        selector.geometry("800x700")

        # Main container with scrollbar
        main_container = EnhancedScrollFrame(selector)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Frame selection controls
        control_frame = ttk.Frame(main_container.interior)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Select reference frame:").pack(side=tk.LEFT)

        # Frame navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.RIGHT)

        self.current_frame_idx = 0
        self.max_frame_idx = len(self.video_frames) - 1

        ttk.Button(nav_frame, text="◄◄", command=lambda: self.navigate_frames(selector, -10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="◄", command=lambda: self.navigate_frames(selector, -1)).pack(side=tk.LEFT, padx=2)

        self.frame_num_entry = ttk.Entry(nav_frame, width=5)
        self.frame_num_entry.pack(side=tk.LEFT, padx=2)
        self.frame_num_entry.insert(0, "1")
        self.frame_num_entry.bind("<Return>", lambda e: self.jump_to_frame(selector))

        ttk.Label(nav_frame, text=f"/ {len(self.video_frames)}").pack(side=tk.LEFT, padx=2)

        ttk.Button(nav_frame, text="►", command=lambda: self.navigate_frames(selector, 1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="►►", command=lambda: self.navigate_frames(selector, 10)).pack(side=tk.LEFT, padx=2)

        # Frame slider
        slider_frame = ttk.Frame(main_container.interior)
        slider_frame.pack(fill=tk.X, pady=5)

        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=self.max_frame_idx,
                                      orient=tk.HORIZONTAL,
                                      command=lambda x: self.update_frame_preview(selector, int(float(x))))
        self.frame_slider.pack(fill=tk.X, expand=True)

        # Vertical slider
        self.vertical_slider = ttk.Scale(main_container.interior, from_=0, to=self.max_frame_idx,
                                         orient=tk.VERTICAL,
                                         command=lambda y: self.update_frame_preview(selector, int(float(y))))
        self.vertical_slider.pack(side=tk.LEFT, fill=tk.Y)

        # Frame info
        self.frame_info = ttk.Label(main_container.interior, text=f"Frame 1 of {len(self.video_frames)}")
        self.frame_info.pack()

        # Preview canvas
        canvas_frame = ttk.Frame(main_container.interior)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(canvas_frame, width=640, height=480, bg='black')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Action buttons
        btn_frame = ttk.Frame(main_container.interior)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Cancel", command=selector.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select and Display",
                   command=lambda: self.select_and_display_frame(selector)).pack(side=tk.RIGHT, padx=5)

        # Show first frame
        self.update_frame_preview(selector, 0)
        self.frame_slider.set(0)
        self.vertical_slider.set(0)

    def update_frame_preview(self, selector, frame_idx):
        """Update the frame preview display"""
        if 0 <= frame_idx < len(self.video_frames):
            self.current_frame_idx = frame_idx
            frame = self.video_frames[frame_idx][1]

            img_pil = Image.fromarray(frame)
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()

            if canvas_width > 0 and canvas_height > 0:
                img_aspect = img_pil.width / img_pil.height
                canvas_aspect = canvas_width / canvas_height

                if img_aspect > canvas_aspect:
                    new_width = canvas_width
                    new_height = int(canvas_width / img_aspect)
                else:
                    new_height = canvas_height
                    new_width = int(canvas_height * img_aspect)

                img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)

            img_tk = ImageTk.PhotoImage(img_pil)
            self.preview_canvas.delete("all")
            self.preview_canvas.img_tk = img_tk
            self.preview_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=img_tk)

            self.frame_info.config(text=f"Frame {frame_idx + 1} of {len(self.video_frames)}")
            self.frame_num_entry.delete(0, tk.END)
            self.frame_num_entry.insert(0, str(frame_idx + 1))

            self.frame_slider.set(frame_idx)
            self.vertical_slider.set(frame_idx)

    def select_and_display_frame(self, selector):
        """Set the selected frame as reference and display in main window"""
        frame_idx = self.current_frame_idx
        if 0 <= frame_idx < len(self.video_frames):
            self.selected_frame_index = frame_idx
            self.original_image = self.video_frames[frame_idx][1].copy()
            self.current_image = self.video_frames[frame_idx][1]
            self.training_image_path = f"video_frame_{frame_idx}"

            self.display_preview()
            self.initialize_label_mask()

            selector.destroy()
            self.status_var.set(f"Selected frame {frame_idx + 1} of {len(self.video_frames)} as reference")
            self.current_step = 1
            self.update_ui_state()

    def navigate_frames(self, selector, delta):
        """Navigate through frames with buttons"""
        new_idx = self.current_frame_idx + delta
        new_idx = max(0, min(new_idx, self.max_frame_idx))
        self.update_frame_preview(selector, new_idx)
        self.frame_slider.set(new_idx)
        self.vertical_slider.set(new_idx)
        self.frame_num_entry.delete(0, tk.END)
        self.frame_num_entry.insert(0, str(new_idx + 1))

    def jump_to_frame(self, selector):
        """Jump to specific frame number"""
        try:
            frame_num = int(self.frame_num_entry.get())
            if 1 <= frame_num <= len(self.video_frames):
                new_idx = frame_num - 1
                self.update_frame_preview(selector, new_idx)
                self.frame_slider.set(new_idx)
                self.vertical_slider.set(new_idx)
            else:
                messagebox.showwarning("Warning", f"Please enter a frame number between 1 and {len(self.video_frames)}")
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid frame number")

    def load_image(self):
        """Load and display the selected image"""
        try:
            img = cv2.imread(self.input_path)
            if img is None:
                raise ValueError("Failed to load image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = img.copy()
            self.current_image = img
            self.training_image_path = self.input_path
            self.display_preview()
            self.initialize_label_mask()
            self.status_var.set(f"Loaded: {os.path.basename(self.input_path)}")
            self.current_step = 1
            self.update_ui_state()
            self.colony_preview_btn.config(state=tk.NORMAL)
        except Exception as e:
            logging.error(f"Image loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def load_image_folder(self):
        """Load images from selected folder"""
        try:
            self.image_files = [f for f in os.listdir(self.input_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if not self.image_files:
                raise ValueError("No images found in folder")

            self.image_files = sorted(self.image_files)
            self.image_selector['values'] = self.image_files
            self.image_selector.current(0)
            self.image_selection_frame.pack(fill=tk.X, pady=5)
            self.status_var.set("Please select training image from folder")
            self.colony_preview_btn.config(state=tk.NORMAL)

        except Exception as e:
            logging.error(f"Folder loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load folder: {str(e)}")

    def load_selected_folder_image(self):
        """Load the selected image from folder"""
        try:
            selected_image = self.image_selector.get()
            img_path = os.path.join(self.input_path, selected_image)
            frame = cv2.imread(img_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {selected_image}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_image = frame.copy()
            self.current_image = frame
            self.training_image_path = img_path
            self.display_preview()
            self.initialize_label_mask()
            self.status_var.set(f"Loaded: {selected_image}")
            self.current_step = 1
            self.update_ui_state()

        except Exception as e:
            logging.error(f"Image loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def show_colony_growth_preview(self):
        """Show a preview of colony growth over time"""
        if not self.video_frames and not self.image_files:
            messagebox.showwarning("Warning", "No frames or images loaded")
            return

        preview_win = tk.Toplevel(self.root)
        preview_win.title("Colony Growth Preview")
        preview_win.geometry("800x700")

        # Main container with scrollbar
        main_container = EnhancedScrollFrame(preview_win)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Vertical slider
        self.colony_vertical_slider = ttk.Scale(main_container.interior, from_=0, to=100, orient=tk.VERTICAL,
                                                command=lambda y: self.update_colony_preview(canvas, int(float(y))))
        self.colony_vertical_slider.pack(side=tk.LEFT, fill=tk.Y)

        # Main canvas
        canvas = tk.Canvas(main_container.interior, bg='white')
        canvas.pack(fill=tk.BOTH, expand=True)

        # Create slider for navigation
        slider_frame = ttk.Frame(main_container.interior)
        slider_frame.pack(fill=tk.X, pady=5)

        self.colony_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                       command=lambda x: self.update_colony_preview(canvas, int(float(x))))
        self.colony_slider.pack(fill=tk.X, padx=10)

        # Frame info
        self.colony_frame_info = ttk.Label(slider_frame, text="Frame 1 of 1")
        self.colony_frame_info.pack()

        # Buttons
        btn_frame = ttk.Frame(main_container.interior)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Play", command=lambda: self.play_colony_preview(canvas)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Stop", command=self.stop_colony_preview).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Close", command=preview_win.destroy).pack(side=tk.RIGHT)

        # Initialize preview
        self.colony_preview_playing = False
        max_frames = len(self.video_frames) if self.video_frames else len(self.image_files)
        self.colony_slider.config(to=max_frames - 1)
        self.colony_vertical_slider.config(to=max_frames - 1)
        self.update_colony_preview(canvas, 0)

    def update_colony_preview(self, canvas, idx):
        """Update the colony growth preview display"""
        if not self.video_frames and not self.image_files:
            return

        if self.video_frames:
            max_idx = len(self.video_frames) - 1
            idx = min(idx, max_idx)
            frame = self.video_frames[idx][1]
            self.colony_frame_info.config(text=f"Frame {idx + 1} of {len(self.video_frames)}")
        else:
            max_idx = len(self.image_files) - 1
            idx = min(idx, max_idx)
            img_path = os.path.join(self.input_path, self.image_files[idx])
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.colony_frame_info.config(text=f"Image {idx + 1} of {len(self.image_files)}")

        img_pil = Image.fromarray(frame)
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width > 0 and canvas_height > 0:
            img_aspect = img_pil.width / img_pil.height
            canvas_aspect = canvas_width / canvas_height

            if img_aspect > canvas_aspect:
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_aspect)

            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.delete("all")
        canvas.img_tk = img_tk
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=img_tk)

        self.colony_slider.set(idx)
        self.colony_vertical_slider.set(idx)

    def play_colony_preview(self, canvas):
        """Animate the colony growth preview"""
        if self.colony_preview_playing:
            return

        self.colony_preview_playing = True
        max_frames = len(self.video_frames) if self.video_frames else len(self.image_files)

        def animate(frame_idx):
            if not self.colony_preview_playing or frame_idx >= max_frames:
                self.colony_preview_playing = False
                return

            self.update_colony_preview(canvas, frame_idx)
            self.root.after(200, lambda: animate(frame_idx + 1))

        animate(0)

    def stop_colony_preview(self):
        """Stop the colony growth animation"""
        self.colony_preview_playing = False

    def display_preview(self):
        """Display preview of current image with optional overlay"""
        if self.current_image is None:
            return

        img_pil = Image.fromarray(self.current_image)

        if self.label_mask is not None and np.any(self.label_mask > 0):
            overlay = np.zeros_like(self.current_image)
            for label, color in self.label_colors.items():
                mask = self.label_mask == label
                overlay[mask] = color

            blended = cv2.addWeighted(self.current_image, 0.7, overlay, 0.3, 0)
            img_pil = Image.fromarray(blended)

        if self.zoom_level != 1.0:
            new_width = int(img_pil.width * self.zoom_level)
            new_height = int(img_pil.height * self.zoom_level)
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.img_tk = img_tk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        self.img_width = self.current_image.shape[1]
        self.img_height = self.current_image.shape[0]

    def initialize_label_mask(self):
        """Initialize the label mask for interactive labeling"""
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            self.label_mask = np.zeros((height, width), dtype=np.uint8)

    def adjust_zoom(self, factor):
        """Adjust zoom level"""
        self.zoom_level *= factor
        self.display_preview()

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_level = 1.0
        self.display_preview()

    def fit_to_window(self):
        """Fit image to window while maintaining aspect ratio"""
        if self.current_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_aspect = self.current_image.shape[1] / self.current_image.shape[0]
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect:
            self.zoom_level = canvas_width / self.current_image.shape[1]
        else:
            self.zoom_level = canvas_height / self.current_image.shape[0]

        self.display_preview()

    def toggle_crop(self):
        """Enable/disable crop functionality"""
        if self.crop_var.get() == 1:
            self.crop_btn.config(state=tk.NORMAL)
            self.status_var.set("Cropping enabled - click 'Select Area' to define crop")
        else:
            self.crop_btn.config(state=tk.DISABLED)
            self.crop_coords = None
            self.crop_mode = False
            self.status_var.set("Cropping disabled")
            if self.original_image is not None:
                self.current_image = self.original_image.copy()
                self.display_preview()
                self.initialize_label_mask()

    def start_crop_selection(self):
        """Start the crop selection process"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        self.crop_mode = True
        self.crop_points = []
        self.status_var.set("Click and drag to select crop area (release to confirm)")

        self.canvas.bind("<Button-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)

    def on_crop_start(self, event):
        """Handle start of crop selection"""
        if not self.crop_mode:
            return

        if self.crop_rect_id:
            self.canvas.delete(self.crop_rect_id)

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        self.crop_start_x = canvas_x
        self.crop_start_y = canvas_y

        self.crop_rect_id = self.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y,
            outline='red', width=2, dash=(5, 5)
        )

    def on_crop_drag(self, event):
        """Handle dragging during crop selection"""
        if not self.crop_mode or self.crop_rect_id is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        self.canvas.coords(self.crop_rect_id,
                           self.crop_start_x, self.crop_start_y,
                           canvas_x, canvas_y)

    def on_crop_end(self, event):
        """Handle end of crop selection"""
        if not self.crop_mode or self.crop_rect_id is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        img_x1 = int(self.crop_start_x / self.zoom_level)
        img_y1 = int(self.crop_start_y / self.zoom_level)
        img_x2 = int(canvas_x / self.zoom_level)
        img_y2 = int(canvas_y / self.zoom_level)

        img_x1 = max(0, min(img_x1, self.img_width))
        img_y1 = max(0, min(img_y1, self.img_height))
        img_x2 = max(0, min(img_x2, self.img_width))
        img_y2 = max(0, min(img_y2, self.img_height))

        x1, x2 = sorted([img_x1, img_x2])
        y1, y2 = sorted([img_y1, img_y2])

        if x2 - x1 > 10 and y2 - y1 > 10:
            self.crop_coords = (x1, y1, x2, y2)
            self.current_image = self.original_image[y1:y2, x1:x2].copy()
            self.initialize_label_mask()
            self.display_preview()
            self.status_var.set(f"Cropped to {x2 - x1}x{y2 - y1}")
            self.current_step = 2
            self.update_ui_state()

        self.canvas.delete(self.crop_rect_id)
        self.crop_rect_id = None
        self.crop_mode = False

        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.bind("<B1-Motion>", self.paint_label)
        self.canvas.bind("<Button-1>", self.paint_label)

    def reset_crop(self):
        """Reset the crop to original image"""
        self.crop_var.set(0)
        self.toggle_crop()
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_preview()
            self.initialize_label_mask()

    def clear_labels(self):
        """Clear all interactive labels"""
        if self.label_mask is not None:
            self.label_mask.fill(0)
            self.display_preview()

    def reset_enhancements(self):
        """Reset all image enhancements to default values"""
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.sharpness_var.set(1.0)
        self.denoise_var.set(0)
        self.adjust_image_quality()

    def adjust_image_quality(self):
        """Adjust image brightness, contrast, and sharpness"""
        if self.current_image is None:
            return

        try:
            img_pil = Image.fromarray(self.current_image)

            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(self.brightness_var.get())

            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(self.contrast_var.get())

            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(self.sharpness_var.get())

            if self.denoise_var.get():
                img_pil = img_pil.filter(ImageFilter.MedianFilter(size=3))

            self.current_image = np.array(img_pil)
            self.display_preview()

        except Exception as e:
            logging.error(f"Image adjustment failed: {str(e)}")

    def set_tool(self, tool):
        """Set the current labeling tool"""
        self.current_tool = tool
        if tool == "brush":
            self.status_var.set("Brush tool selected")
        elif tool == "erase":
            self.status_var.set("Eraser tool selected")

    def paint_label(self, event):
        """Handle painting labels on the image"""
        if self.current_image is None or self.label_mask is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        img_x = int(canvas_x / self.zoom_level)
        img_y = int(canvas_y / self.zoom_level)

        img_x = max(0, min(img_x, self.img_width - 1))
        img_y = max(0, min(img_y, self.img_height - 1))

        brush_size = self.brush_size.get()
        half_size = brush_size // 2

        if self.current_tool == "brush":
            label_value = self.current_label
        elif self.current_tool == "erase":
            label_value = 0
        else:
            return

        for y in range(max(0, img_y - half_size), min(self.img_height, img_y + half_size + 1)):
            for x in range(max(0, img_x - half_size), min(self.img_width, img_x + half_size + 1)):
                dist = np.sqrt((x - img_x) ** 2 + (y - img_y) ** 2)
                if dist <= half_size:
                    self.label_mask[y, x] = label_value

        self.display_preview()

    def reset_last_coords(self, event):
        """Reset the last coordinates for painting"""
        self.last_x = None
        self.last_y = None

    def train_classifier(self):
        """Train random forest classifier on labeled pixels"""
        if self.current_image is None or self.label_mask is None or np.all(self.label_mask == 0):
            messagebox.showwarning("Warning", "Please label some pixels first")
            return

        try:
            self.status_var.set("Training classifier...")
            self.root.update()

            self.reference_image = self.current_image.copy()
            features = self.extract_features(self.reference_image)

            X = []
            y = []

            for i in range(self.current_image.shape[0]):
                for j in range(self.current_image.shape[1]):
                    if self.label_mask[i, j] > 0:
                        pixel_features = []
                        for feature_type, feature_data in features.items():
                            if isinstance(feature_data, np.ndarray):
                                if len(feature_data.shape) == 2:
                                    pixel_features.append(feature_data[i, j])
                                else:
                                    pixel_features.extend(feature_data[i, j])

                        X.append(pixel_features)
                        y.append(self.label_mask[i, j])

            if len(X) == 0:
                raise ValueError("No labeled pixels found")

            if self.training_data is None:
                self.training_data = np.array(X)
                self.training_labels = np.array(y)
            else:
                self.training_data = np.vstack((self.training_data, X))
                self.training_labels = np.concatenate((self.training_labels, y))

            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(self.training_data, self.training_labels)

            messagebox.showinfo("Training Complete", "Classifier trained successfully")
            self.status_var.set("Classifier trained")
            self.current_step = 3
            self.update_ui_state()

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def extract_features(self, img):
        """Extract features based on current selection"""
        features = {}
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Gaussian Smoothing
        if "Gaussian Smoothing" in self.feature_params and self.feature_params["Gaussian Smoothing"]["var"].get():
            for i, sigma_var in enumerate(self.sigma_vars):
                sigma = sigma_var.get()
                if sigma > 0:
                    features[f"Gaussian_{i}"] = gaussian(gray, sigma=sigma)

        # Edge detection
        if "Edge" in self.feature_params and self.feature_params["Edge"]["var"].get():
            gx = sobel(gray, axis=0)
            gy = sobel(gray, axis=1)
            features["Edge"] = np.stack([gx, gy], axis=-1)

        # Laplacian of Gaussian
        if "Laplacian of Gaussian" in self.feature_params and self.feature_params["Laplacian of Gaussian"]["var"].get():
            for i, sigma_var in enumerate(self.sigma_vars):
                sigma = sigma_var.get()
                if sigma > 0:
                    features[f"LoG_{i}"] = laplace(gaussian(gray, sigma=sigma))

        # Gaussian Gradient Magnitude
        if "Gaussian Gradient Magnitude" in self.feature_params and self.feature_params["Gaussian Gradient Magnitude"][
            "var"].get():
            for i, sigma_var in enumerate(self.sigma_vars):
                sigma = sigma_var.get()
                if sigma > 0:
                    gx = gaussian(gray, sigma=sigma, order=[0, 1])
                    gy = gaussian(gray, sigma=sigma, order=[1, 0])
                    features[f"GGM_{i}"] = np.sqrt(gx ** 2 + gy ** 2)

        # Difference of Gaussians
        if "Difference of Gaussians" in self.feature_params and self.feature_params["Difference of Gaussians"][
            "var"].get():
            for i in range(len(self.sigma_vars) - 1):
                sigma1 = self.sigma_vars[i].get()
                sigma2 = self.sigma_vars[i + 1].get()
                if sigma1 > 0 and sigma2 > 0:
                    g1 = gaussian(gray, sigma=sigma1)
                    g2 = gaussian(gray, sigma=sigma2)
                    features[f"DoG_{i}"] = g1 - g2

        # Texture features
        if "Texture" in self.feature_params and self.feature_params["Texture"]["var"].get():
            for i, sigma_var in enumerate(self.sigma_vars):
                sigma = sigma_var.get()
                if sigma > 0:
                    filt_real, filt_imag = gabor(gray, frequency=0.6, sigma_x=sigma, sigma_y=sigma)
                    features[f"Gabor_{i}"] = np.sqrt(filt_real ** 2 + filt_imag ** 2)

        # Structure Tensor Eigenvalues
        if "Structure Tensor Eigenvalues" in self.feature_params and \
                self.feature_params["Structure Tensor Eigenvalues"]["var"].get():
            gx = sobel(gray, axis=0)
            gy = sobel(gray, axis=1)
            gxx = gaussian(gx * gx, sigma=1)
            gxy = gaussian(gx * gy, sigma=1)
            gyy = gaussian(gy * gy, sigma=1)

            lambda1 = 0.5 * (gxx + gyy + np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2))
            lambda2 = 0.5 * (gxx + gyy - np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2))

            features["Structure Tensor"] = np.stack([lambda1, lambda2], axis=-1)

        # Hessian of Gaussian Eigenvalue
        if "Hessian of Gaussian Eigenvalue" in self.feature_params and \
                self.feature_params["Hessian of Gaussian Eigenvalue"]["var"].get():
            for i, sigma_var in enumerate(self.sigma_vars):
                sigma = sigma_var.get()
                if sigma > 0:
                    H = hessian_matrix(gray, sigma=sigma, order='rc')
                    eigenvalues = hessian_matrix_eigvals(H)
                    features[f"Hessian_{i}"] = eigenvalues[0]

        return features

    def apply_features(self):
        """Apply selected features and update display"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        try:
            self.status_var.set("Applying features...")
            self.root.update()

            features = self.extract_features(self.current_image)

            if features:
                first_feature = next(iter(features.values()))
                if isinstance(first_feature, np.ndarray):
                    if len(first_feature.shape) == 2:
                        norm_feature = cv2.normalize(first_feature, None, 0, 255, cv2.NORM_MINMAX)
                        display_img = cv2.cvtColor(norm_feature.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    else:
                        norm_feature = cv2.normalize(first_feature[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
                        display_img = cv2.cvtColor(norm_feature.astype(np.uint8), cv2.COLOR_GRAY2RGB)

                    self.current_image = display_img
                    self.display_preview()

            self.status_var.set("Features applied")

        except Exception as e:
            logging.error(f"Feature application failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply features: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def process_current_image(self):
        """Process only the current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        if self.classifier is None:
            messagebox.showwarning("Warning", "Please train the classifier first")
            return

        try:
            self.status_var.set("Processing current image...")
            self.progress['value'] = 0
            self.root.update()

            features = self.extract_features(self.current_image)
            segmented = self.segment_image(self.current_image, features)

            if self.binary_output_var.get():
                segmented = self.convert_to_binary(segmented)

            output_path = os.path.join(self.output_folder, "segmented.png")
            cv2.imwrite(output_path, segmented)

            self.processed_image = segmented
            self.show_result(segmented)

            self.progress['value'] = 100
            self.status_var.set("Processing completed")

        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def segment_image(self, img, features):
        """Perform segmentation using extracted features and trained classifier"""
        try:
            height, width = img.shape[:2]
            X = []

            for i in range(height):
                for j in range(width):
                    pixel_features = []
                    for feature_type, feature_data in features.items():
                        if isinstance(feature_data, np.ndarray):
                            if len(feature_data.shape) == 2:
                                pixel_features.append(feature_data[i, j])
                            else:
                                pixel_features.extend(feature_data[i, j])
                    X.append(pixel_features)

            labels = self.classifier.predict(X)
            labels = labels.reshape((height, width))

            return labels

        except Exception as e:
            logging.error(f"Segmentation failed: {str(e)}")
            raise ValueError(f"Segmentation error: {str(e)}")

    def convert_to_binary(self, segmented):
        """Convert segmented image to binary mask"""
        binary_mask = np.zeros_like(segmented)
        binary_mask[segmented == 1] = 255
        binary_mask[segmented == 2] = 0
        binary_mask[segmented == 3] = 255
        binary_mask[segmented == 4] = 255
        return binary_mask

    def show_result(self, segmented):
        """Display segmentation result"""
        if len(segmented.shape) == 2:
            display_img = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
        else:
            display_img = segmented.copy()

        self.current_image = display_img
        self.display_preview()

    def preview_segmentation(self):
        """Preview the segmentation result"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        if self.classifier is None:
            messagebox.showwarning("Warning", "Please train the classifier first")
            return

        try:
            self.status_var.set("Previewing segmentation...")
            self.root.update()

            features = self.extract_features(self.current_image)
            segmented = self.segment_image(self.current_image, features)

            if self.binary_output_var.get():
                segmented = self.convert_to_binary(segmented)

            self.show_result(segmented)
            self.status_var.set("Segmentation preview complete")

        except Exception as e:
            logging.error(f"Preview failed: {str(e)}")
            messagebox.showerror("Error", f"Preview failed: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def toggle_feature_suggestion(self):
        """Toggle feature suggestion mode"""
        if self.suggest_features_var.get():
            self.status_var.set("Feature suggestion enabled")
        else:
            self.status_var.set("Feature suggestion disabled")

    def toggle_live_update(self):
        """Toggle live update mode"""
        if self.live_update_var.get():
            self.status_var.set("Live update enabled")
        else:
            self.status_var.set("Live update disabled")

    def add_label(self):
        """Add a new custom label"""
        label_name = simpledialog.askstring("Add Label", "Enter label name:", parent=self.root)
        if label_name:
            hue = (self.next_label_id * 0.618) % 1.0
            r, g, b = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.8)]

            self.label_colors[self.next_label_id] = (r, g, b)
            self.label_names[self.next_label_id] = label_name
            self.next_label_id += 1

            self.update_label_preview()
            self.status_var.set(f"Added new label: {label_name}")

    def update_label_preview(self):
        """Update the lab