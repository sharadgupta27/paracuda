import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import multiprocessing
from datetime import datetime
import os
import traceback
import joblib
import rasterio
import socket
import sys
import webbrowser

# Import our modular components
from data_processing import safe_interpolate_spectra, preprocess_spectra, calculate_statistics, filter_wavelengths
from model_training import create_model, optimize_components_parallel, parse_parameter_value
from cross_validation import perform_cross_validation
from image_processing import process_image_for_prediction, save_prediction_image
from file_operations import save_results_to_excel, generate_default_filename, generate_model_filename

class SpectralAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.iconbitmap('icon.ico')
        self.title("Paracuda III")
        self.geometry("850x800")
        self.resizable(False, False)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create menu bar
        self.create_menu()
        
        # Initialize optimize components variable
        self.optimize_components_var = tk.BooleanVar()
        
        # Data storage
        self.df = None
        self.wavelengths = None
        self.soil_properties = None
        self.input_filename = None
        self.output_filename = None
        self.trained_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.image_data = None
        self.image_meta = None
        self.predicted_image = None
        self.image_canvas = None
        self.selected_property = None
        self.filtered_wavelengths = None
        self.new_wavelengths = None
        
        # Available colormaps
        self.colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                         'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        
        # Model parameters configuration with XGBoost
        self.model_params = {
            "PLS-R": {
                "params": {
                    "n_components": {
                        "label": "Components", 
                        "default": "32", 
                        "type": "entry",
                        "help": "Number of components to keep"
                    }
                }
            },
            "SVM": {
                "params": {
                    "kernel": {
                        "label": "Kernel", 
                        "default": "rbf", 
                        "type": "combobox",
                        "values": ["linear", "poly", "rbf", "sigmoid"],
                        "help": "Specifies the kernel type to be used"
                    },
                    "C": {
                        "label": "Regularization (C)", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Regularization parameter"
                    },
                    "degree": {
                        "label": "Degree", 
                        "default": "3", 
                        "type": "entry",
                        "help": "Degree of polynomial kernel (ignored by other kernels)"
                    },
                    "gamma": {
                        "label": "Gamma", 
                        "default": "scale", 
                        "type": "combobox",
                        "values": ["scale", "auto"],
                        "help": "Kernel coefficient for rbf, poly and sigmoid"
                    },
                    "epsilon": {
                        "label": "Epsilon", 
                        "default": "0.1", 
                        "type": "entry",
                        "help": "Epsilon in the epsilon-SVR model"
                    }
                }
            },
            "Ridge": {
                "params": {
                    "alpha": {
                        "label": "Alpha", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Regularization strength"
                    },
                    "solver": {
                        "label": "Solver", 
                        "default": "auto", 
                        "type": "combobox",
                        "values": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                        "help": "Solver to use in the computational routines"
                    },
                    "max_iter": {
                        "label": "Max Iterations", 
                        "default": "1000", 
                        "type": "entry",
                        "help": "Maximum number of iterations"
                    }
                }
            },
            "Lasso": {
                "params": {
                    "alpha": {
                        "label": "Alpha", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Constant that multiplies the L1 term"
                    },
                    "max_iter": {
                        "label": "Max Iterations", 
                        "default": "1000", 
                        "type": "entry",
                        "help": "Maximum number of iterations"
                    },
                    "selection": {
                        "label": "Selection", 
                        "default": "cyclic", 
                        "type": "combobox",
                        "values": ["cyclic", "random"],
                        "help": "Selection method for updating coefficients"
                    },
                    "tol": {
                        "label": "Tolerance", 
                        "default": "1e-4", 
                        "type": "entry",
                        "help": "Tolerance for the optimization"
                    }
                }
            },
            "PCA": {
                "params": {
                    "n_components": {
                        "label": "Components", 
                        "default": "10", 
                        "type": "entry",
                        "help": "Number of components to keep"
                    },
                    "svd_solver": {
                        "label": "SVD Solver", 
                        "default": "auto", 
                        "type": "combobox",
                        "values": ["auto", "full", "arpack", "randomized"],
                        "help": "SVD solver to use"
                    },
                    "whiten": {
                        "label": "Whiten", 
                        "default": "False", 
                        "type": "combobox",
                        "values": ["True", "False"],
                        "help": "When True, components are divided by sqrt(n_samples)"
                    }
                }
            },
            "Linear": {
                "params": {
                    "fit_intercept": {
                        "label": "Fit Intercept", 
                        "default": "True", 
                        "type": "combobox",
                        "values": ["True", "False"],
                        "help": "Whether to calculate the intercept"
                    }
                }
            },
            "Random Forest": {
                "params": {
                    "n_estimators": {
                        "label": "Number of Trees", 
                        "default": "100", 
                        "type": "entry",
                        "help": "Number of trees in the forest"
                    },
                    "max_depth": {
                        "label": "Max Depth", 
                        "default": "5", 
                        "type": "entry",
                        "help": "Maximum depth of the tree"
                    },
                    "min_samples_split": {
                        "label": "Min Samples Split", 
                        "default": "2", 
                        "type": "entry",
                        "help": "Minimum samples required to split an internal node"
                    },
                    "min_samples_leaf": {
                        "label": "Min Samples Leaf", 
                        "default": "1",
                        "type": "entry",
                        "help": "Minimum samples required to be at a leaf node"
                    },
                    "max_features": {
                        "label": "Max Features", 
                        "default": "sqrt",
                        "type": "combobox",
                        "values": ["sqrt", "log2", "None"],
                        "help": "Number of features to consider when looking for best split"
                    },
                    "bootstrap": {
                        "label": "Bootstrap", 
                        "default": "True", 
                        "type": "combobox",
                        "values": ["True", "False"],
                        "help": "Whether bootstrap samples are used when building trees"
                    }
                }
            },
            "XGBoost": {
                "params": {
                    "n_estimators": {
                        "label": "N Estimators", 
                        "default": "100", 
                        "type": "entry",
                        "help": "Number of boosting rounds"
                    },
                    "max_depth": {
                        "label": "Max Depth", 
                        "default": "6", 
                        "type": "entry",
                        "help": "Maximum depth of trees"
                    },
                    "learning_rate": {
                        "label": "Learning Rate", 
                        "default": "0.3", 
                        "type": "entry",
                        "help": "Step size shrinkage used to prevent overfitting"
                    },
                    "subsample": {
                        "label": "Subsample", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Subsample ratio of training instances"
                    },
                    "colsample_bytree": {
                        "label": "Col Sample Tree", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "Subsample ratio of columns when constructing each tree"
                    },
                    "reg_alpha": {
                        "label": "Reg Alpha", 
                        "default": "0.0", 
                        "type": "entry",
                        "help": "L1 regularization term on weights"
                    },
                    "reg_lambda": {
                        "label": "Reg Lambda", 
                        "default": "1.0", 
                        "type": "entry",
                        "help": "L2 regularization term on weights"
                    }
                }
            }

        }
        
        self.param_vars = {}
        self.param_widgets = {}
        self.cv_param_vars = {}
        self.cv_param_widgets = {}
        
        self.create_gui()
        self.update_datetime()
    
    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_help)
    
    def show_help(self):
        try:
            # Get the path to the help PDF
            if getattr(sys, 'frozen', False):
                # If running as executable
                help_path = os.path.join(sys._MEIPASS, 'help_documentation.pdf')
            else:
                # If running as script
                help_path = 'help_documentation.pdf'
            
            # Open the PDF file with the default PDF viewer
            if os.path.exists(help_path):
                if sys.platform.startswith('darwin'):  # macOS
                    os.system(f'open "{help_path}"')
                elif sys.platform.startswith('win'):  # Windows
                    os.startfile(help_path)
                else:  # Linux
                    os.system(f'xdg-open "{help_path}"')
            else:
                messagebox.showerror("Error", "Help documentation not found!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open help documentation: {str(e)}")
    
    def update_datetime(self):
        current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        computer_name = socket.gethostname()
        self.datetime_label.config(text=f"{current_time}\nComputer: {computer_name}")
        self.after(1000, self.update_datetime)
        
    def create_gui(self):
        main_frame = ttk.Frame(self, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Title and DateTime
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        title_frame.grid_columnconfigure(1, weight=1)
        
        title_label = ttk.Label(title_frame, text="Paracuda", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=5)
        
        self.datetime_label = ttk.Label(title_frame, text="", font=('Helvetica', 10))
        self.datetime_label.grid(row=0, column=1, pady=5, sticky=tk.E)
        
        # Loading Files Section (Row 0)
        loading_frame = ttk.LabelFrame(main_frame, text="Loading Files", padding="5")
        loading_frame.grid(row=1, column=0, columnspan=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        loading_frame.grid_columnconfigure(0, weight=1)
        loading_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Button(loading_frame, text="Load Excel File", 
                  command=self.load_excel).grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(loading_frame, text="Check Excel", 
                  command=self.check_excel).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Left Column - Row 1: What To Do + Soil Property Selection
        left_col_row1 = ttk.Frame(main_frame)
        left_col_row1.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=5, padx=5)
        left_col_row1.grid_columnconfigure(0, weight=1)
        
        # What To Do Section
        what_to_do_frame = ttk.LabelFrame(left_col_row1, text="What To Do", padding="5")
        what_to_do_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        what_to_do_frame.grid_columnconfigure(0, weight=1)
        
        self.develop_models_var = tk.BooleanVar()
        ttk.Checkbutton(what_to_do_frame, text="Develop Prediction Models", 
                       variable=self.develop_models_var).grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.apply_models_var = tk.BooleanVar()
        ttk.Checkbutton(what_to_do_frame, text="Apply Models on Image", 
                       variable=self.apply_models_var,
                       command=self.toggle_image_options).grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.export_stats_var = tk.BooleanVar()
        ttk.Checkbutton(what_to_do_frame, text="Export Stats Excel File", 
                       variable=self.export_stats_var).grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # Soil Property Selection
        property_frame = ttk.LabelFrame(left_col_row1, text="Soil Property Selection", padding="5")
        property_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        property_frame.grid_columnconfigure(0, weight=1)
        ttk.Button(property_frame, text="Select Soil Property", command=self.select_soil_property).grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Left Column - Row 2: Resource Allocation + Spectral Configuration
        left_col_row2 = ttk.Frame(main_frame)
        left_col_row2.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N), pady=5, padx=5)
        left_col_row2.grid_columnconfigure(0, weight=1)
        
        # Resource Allocation
        resource_frame = ttk.LabelFrame(left_col_row2, text="Resource Allocation", padding="5")
        resource_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        resource_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(resource_frame, text="Number of Cores to Use").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.cores_var = tk.StringVar(value="1")
        ttk.Combobox(resource_frame, textvariable=self.cores_var,
                     values=list(range(1, multiprocessing.cpu_count() + 1)), width=25, state='readonly').grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Spectral Configuration
        spectral_frame = ttk.LabelFrame(left_col_row2, text="Spectral Configuration", padding="5")
        spectral_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        spectral_frame.grid_columnconfigure(0, weight=1)
        spectral_frame.grid_columnconfigure(1, weight=1)
        spectral_frame.grid_columnconfigure(2, weight=1)
        spectral_frame.grid_columnconfigure(3, weight=1)

        # Resampling and Spacing
        ttk.Label(spectral_frame, text="Resampling:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.resampling_var = tk.StringVar(value="No")
        ttk.Combobox(spectral_frame, textvariable=self.resampling_var, values=["Yes", "No"], width=7, state='readonly').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        ttk.Label(spectral_frame, text="Spacing (nm):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.spacing_var = tk.StringVar(value="10")
        ttk.Entry(spectral_frame, textvariable=self.spacing_var, width=8).grid(row=0, column=3, sticky=(tk.W, tk.E), padx=5, pady=2)

        # Min Wave and Max Wave
        wavelength_frame = ttk.Frame(spectral_frame)
        wavelength_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E))
        wavelength_frame.grid_columnconfigure(1, weight=1)
        wavelength_frame.grid_columnconfigure(3, weight=1)
        ttk.Label(wavelength_frame, text="Min Wave:").grid(row=0, column=0, padx=5)
        self.min_wave_var = tk.StringVar()
        ttk.Entry(wavelength_frame, textvariable=self.min_wave_var, width=8).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Label(wavelength_frame, text="Max Wave:").grid(row=0, column=2, padx=5)
        self.max_wave_var = tk.StringVar()
        ttk.Entry(wavelength_frame, textvariable=self.max_wave_var, width=8).grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        # Image Processing Section
        image_frame = ttk.LabelFrame(main_frame, text="Apply on Image (Optional)", padding="5")
        image_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        image_frame.grid_columnconfigure(2, weight=1)
        
        self.load_image_btn = ttk.Button(image_frame, text="Load Image File", 
                                       command=self.load_image, state='disabled')
        self.load_image_btn.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        
        self.predict_image_btn = ttk.Button(image_frame, text="Predict on Image", 
                                          command=self.save_prediction, state='disabled')
        self.predict_image_btn.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        self.view_image_btn = ttk.Button(image_frame, text="View Predicted Image", 
                                       command=self.view_predicted_image, state='disabled')
        self.view_image_btn.grid(row=0, column=2, padx=5, sticky=(tk.W, tk.E))
        
        # Colormap selection
        ttk.Label(image_frame, text="Colormap:").grid(row=1, column=0, sticky=tk.W, pady=(5,0))
        self.colormap_var = tk.StringVar(value='viridis')
        self.colormap_combo = ttk.Combobox(image_frame, textvariable=self.colormap_var,
                                         values=self.colormaps, state='disabled')
        self.colormap_combo.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(5,0))
        self.colormap_combo.bind('<<ComboboxSelected>>', self.update_colormap)
        
        # Right Column - Pre-Processing Section
        preprocess_frame = ttk.LabelFrame(main_frame, text="Pre-Processing", padding="5")
        preprocess_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N), pady=5, padx=5)
        preprocess_frame.grid_columnconfigure(0, weight=1)
        
        self.preprocess_var = tk.StringVar(value="No Preprocessing")
        preprocess_options = ["No Preprocessing", "Continuum Removal", "First Derivative", 
                            "Second Derivative", "Absorbance"]
        ttk.Label(preprocess_frame, text="Choose Type").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(preprocess_frame, textvariable=self.preprocess_var, 
                    values=preprocess_options, state='readonly').grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Right Column - Modeling Section
        self.model_frame = ttk.LabelFrame(main_frame, text="Modeling", padding="5")
        self.model_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.model_frame.grid_columnconfigure(0, weight=1)
        
        self.model_var = tk.StringVar(value="PLS-R")
        ttk.Label(self.model_frame, text="Choose Type").grid(row=0, column=0, sticky=tk.W)
        model_options = list(self.model_params.keys())
        model_combo = ttk.Combobox(self.model_frame, textvariable=self.model_var, 
                                 values=model_options, state='readonly')
        model_combo.grid(row=1, column=0, sticky=(tk.W, tk.E))
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Create scrollable frame for parameters
        self.create_scrollable_params_frame()
        
        # Component Optimization checkbox (moved to Modeling frame)
       # self.optimize_components_var = tk.BooleanVar()
       # self.optimize_components_cb = ttk.Checkbutton(self.model_frame, text="Optimize Components", 
       #                variable=self.optimize_components_var)
       # self.optimize_components_cb.grid(row=3, column=0, sticky=tk.W, pady=0)
        
        # Right Column - Validation Strategy Section
        validation_frame = ttk.LabelFrame(main_frame, text="Validation Strategy", padding="5")
        validation_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        validation_frame.grid_columnconfigure(0, weight=1)
        validation_frame.grid_columnconfigure(1, weight=1)
        
        # Train-Test Split
        ttk.Label(validation_frame, text="Train-Test Split:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(validation_frame, textvariable=self.test_size_var, width=10).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Cross-Validation Strategy
        ttk.Label(validation_frame, text="Cross-Validation:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cv_strategy_var = tk.StringVar(value="None")
        cv_strategies = ["None", "K-Fold", "Leave-One-Out", "Leave-P-Out"]
        cv_combo = ttk.Combobox(validation_frame, textvariable=self.cv_strategy_var,
                               values=cv_strategies, state='readonly')
        cv_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        cv_combo.bind('<<ComboboxSelected>>', self.on_cv_strategy_change)
        
        # CV Parameters frame
        self.cv_params_frame = ttk.Frame(validation_frame)
        self.cv_params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        self.cv_params_frame.grid_columnconfigure(0, weight=1)
        self.cv_params_frame.grid_columnconfigure(1, weight=1)
        
        # Model Management + Start Analysis
        right_stack = ttk.Frame(main_frame)
        right_stack.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        right_stack.grid_columnconfigure(0, weight=1)

        # Model Management Section
        self.model_mgmt_frame = ttk.LabelFrame(right_stack, text="Model Management", padding="5")
        self.model_mgmt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        self.model_mgmt_frame.grid_columnconfigure(0, weight=1)
        self.model_mgmt_frame.grid_columnconfigure(1, weight=1)
        self.model_mgmt_frame.grid_columnconfigure(2, weight=1)
        self.save_model_btn = ttk.Button(self.model_mgmt_frame, text="Save Model", command=self.save_model, state='disabled')
        self.save_model_btn.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        self.load_model_btn = ttk.Button(self.model_mgmt_frame, text="Load Model", command=self.load_model, state='disabled')
        self.load_model_btn.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.view_model_btn = ttk.Button(self.model_mgmt_frame, text="View Model", command=self.view_model, state='disabled')
        self.view_model_btn.grid(row=0, column=2, padx=5, sticky=(tk.W, tk.E))
        
        # Start Button
        self.start_button = ttk.Button(right_stack, text="Start Analysis", command=self.start_analysis)
        self.start_button.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Status Messages
        self.status_text = tk.Text(main_frame, height=6, width=50)
        self.status_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.status_text.bind("<Button>", lambda event: self.status_text.focus_set())
        
        # Progress Bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=7, column=0, columnspan=2, pady=2, sticky=(tk.W, tk.E))
        progress_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(progress_frame, text="Progress:").grid(row=0, column=0, padx=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.grid(row=0, column=2, padx=5)
        
        # Copyright Message
        copyright_frame = ttk.LabelFrame(main_frame, padding="2")
        copyright_frame.grid(row=8, column=0, columnspan=2, pady=0, sticky=(tk.W, tk.E))
        main_frame.grid_rowconfigure(7, weight=0)
        main_frame.grid_rowconfigure(8, weight=0)
        copyright_frame.grid_columnconfigure(0, weight=1)
        
        copyright_text = "Developed by Sharad Kumar Gupta (sharad.gupta@ufz.de)\nat Helmholtz Centre for Environmental Research, Leipzig, Germany"
        ttk.Label(copyright_frame, text=copyright_text, justify=tk.CENTER).grid(row=0, column=0)
        
        # Initialize parameter entries for default model
        self.on_model_change(None)
        self.on_cv_strategy_change(None)
        #self.update_optimize_components_visibility()
    
    def create_scrollable_params_frame(self):
        # Remove old frame if exists
        if hasattr(self, 'params_canvas'):
            self.params_canvas.destroy()
        if hasattr(self, 'params_frame'):
            self.params_frame.destroy()
        if hasattr(self, 'params_scrollbar'):
            self.params_scrollbar.destroy()

        # Create frame container for canvas and scrollbar
        self.params_container = ttk.Frame(self.model_frame)
        self.params_container.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(4, 0))
        self.params_container.grid_columnconfigure(0, weight=1)
        self.params_container.grid_rowconfigure(0, weight=1)

        # Canvas for scrollability - dynamic height based on content
        self.params_canvas = tk.Canvas(self.params_container, highlightthickness=0, bd=0)
        self.params_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar
        self.params_scrollbar = ttk.Scrollbar(self.params_container, orient="vertical", command=self.params_canvas.yview)
        self.params_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)

        # Frame inside canvas for parameters
        self.params_frame = ttk.Frame(self.params_canvas)
        self.canvas_window = self.params_canvas.create_window((0, 0), window=self.params_frame, anchor='nw')

        # Populate parameters
        self.param_vars.clear()
        self.param_widgets.clear()
        model = self.model_var.get()
        params = self.model_params[model]["params"]
        
        # Calculate layout - 2 columns of parameter pairs
        row = 0
        col = 0
        max_params_per_row = 2
        
        for pname, pinfo in params.items():
            # Label
            ttk.Label(self.params_frame, text=pinfo.get("label", pname), width=18).grid(
                row=row, column=col*2, sticky=tk.W, padx=2, pady=2)
            
            # Input widget
            if pinfo["type"] == "entry":
                var = tk.StringVar(value=pinfo.get("default", ""))
                widget = ttk.Entry(self.params_frame, textvariable=var, width=12)
            elif pinfo["type"] == "combobox":
                var = tk.StringVar(value=pinfo.get("default", ""))
                widget = ttk.Combobox(self.params_frame, textvariable=var, 
                                    values=pinfo.get("values", []), width=12, state='readonly')
            
            widget.grid(row=row, column=col*2+1, sticky=tk.W, padx=2, pady=2)
            self.param_vars[pname] = var
            self.param_widgets[pname] = widget
            
            # Move to next position
            col += 1
            if col >= max_params_per_row:
                col = 0
                row += 1

        # Add optimize components checkbox for PCA/PLSR
        if model in ["PLS-R", "PCA"]:
            # Add some spacing
            if col > 0:  # If we're not at the start of a new row
                row += 1
                col = 0
            
            ttk.Separator(self.params_frame, orient='horizontal').grid(
                row=row, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
            row += 1
            
            # Reset the variable to ensure it's properly bound
            if not hasattr(self, 'optimize_components_var'):
                self.optimize_components_var = tk.BooleanVar()
            
            ttk.Checkbutton(self.params_frame, text="Optimize Components", 
                           variable=self.optimize_components_var).grid(
                row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        else:
            # Reset to False for non-PCA/PLSR models
            if hasattr(self, 'optimize_components_var'):
                self.optimize_components_var.set(False)

        # Configure scrolling
        def configure_scroll_region(event=None):
            self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
            
            # Calculate required height (max 200px, min 100px)
            bbox = self.params_canvas.bbox("all")
            if bbox:
                content_height = bbox[3] - bbox[1]
                canvas_height = min(max(content_height + 10, 100), 200)
                self.params_canvas.configure(height=canvas_height)
                
                # Show/hide scrollbar based on content
                if content_height > canvas_height - 15:
                    self.params_scrollbar.grid()
                else:
                    self.params_scrollbar.grid_remove()

        def configure_canvas_width(event):
            # Make the frame width match the canvas width
            canvas_width = event.width
            self.params_canvas.itemconfig(self.canvas_window, width=canvas_width)

        # Bind events
        self.params_frame.bind("<Configure>", configure_scroll_region)
        self.params_canvas.bind("<Configure>", configure_canvas_width)
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            self.params_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            self.params_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            self.params_canvas.unbind_all("<MouseWheel>")
        
        self.params_canvas.bind('<Enter>', bind_mousewheel)
        self.params_canvas.bind('<Leave>', unbind_mousewheel)
        
        # Initial configuration
        self.params_frame.update_idletasks()
        configure_scroll_region()
    
    def on_cv_strategy_change(self, event=None):
        # Clear existing CV parameter widgets
        for widget in self.cv_params_frame.winfo_children():
            widget.destroy()
        
        cv_strategy = self.cv_strategy_var.get()
        
        if cv_strategy == "K-Fold":
            ttk.Label(self.cv_params_frame, text="K (folds):").grid(row=0, column=0, sticky=tk.W, padx=2)
            self.k_folds_var = tk.StringVar(value="5")
            ttk.Entry(self.cv_params_frame, textvariable=self.k_folds_var, width=8).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)
            
            ttk.Label(self.cv_params_frame, text="Shuffle:").grid(row=0, column=2, sticky=tk.W, padx=2)
            self.shuffle_var = tk.StringVar(value="True")
            ttk.Combobox(self.cv_params_frame, textvariable=self.shuffle_var, 
                        values=["True", "False"], state='readonly', width=8).grid(row=0, column=3, sticky=(tk.W, tk.E), padx=2)
        
        elif cv_strategy == "Leave-P-Out":
            ttk.Label(self.cv_params_frame, text="P (samples):").grid(row=0, column=0, sticky=tk.W, padx=2)
            self.p_out_var = tk.StringVar(value="2")
            ttk.Entry(self.cv_params_frame, textvariable=self.p_out_var, width=8).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)
    
    def on_model_change(self, event=None):
        self.create_scrollable_params_frame()
        
        # Hide/show optimize components checkbox in validation frame based on model
        model = self.model_var.get()
        if hasattr(self, 'optimize_components_checkbox'):
            if model in ["PLS-R", "PCA"]:
                self.optimize_components_checkbox.grid()
            else:
                self.optimize_components_checkbox.grid_remove()
    
    def toggle_image_options(self):
        state = 'normal' if self.apply_models_var.get() else 'disabled'
        self.load_image_btn.config(state=state)
        self.predict_image_btn.config(state=state)
        self.view_image_btn.config(state=state)
        self.colormap_combo.config(state=state)
    
    def update_progress(self, value, message=""):
        self.progress_var.set(value)
        self.progress_label.config(text=f"{int(value)}%")
        if message:
            self.status_text.insert(tk.END, f"{message}\n")
            self.status_text.see(tk.END)
        self.update_idletasks()
    
    def reset_gui(self):
        self.develop_models_var.set(False)
        self.apply_models_var.set(False)
        self.export_stats_var.set(False)
        self.preprocess_var.set("No Preprocessing")
        self.model_var.set("PLS-R")
        self.cores_var.set("1")
        self.resampling_var.set("No")
        self.spacing_var.set("10")
        self.min_wave_var.set("")
        self.max_wave_var.set("")
        self.colormap_var.set("viridis")
        self.test_size_var.set("0.2")
        self.cv_strategy_var.set("None")
        self.optimize_components_var.set(False)
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self.status_text.delete(1.0, tk.END)
        self.toggle_image_options()
        self.on_model_change(None)
        self.on_cv_strategy_change(None)
        self.save_model_btn.config(state='disabled')
        self.load_model_btn.config(state='disabled')
        self.view_model_btn.config(state='disabled')
    
    def view_predicted_image(self):
        if self.predicted_image is None:
            messagebox.showwarning("Warning", "No predicted image available")
            return
        
        if hasattr(self, 'image_window') and self.image_window.winfo_exists():
            self.image_window.lift()
            return
        
        self.image_window = tk.Toplevel(self)
        self.image_window.title("Predicted Image")
        self.image_window.geometry("600x500")
        
        # Create figure and canvas
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        # Mask values below 0
        masked_image = np.ma.masked_where(self.predicted_image <= 0, self.predicted_image)
        
        # Plot image with current colormap
        im = ax.imshow(masked_image, cmap=self.colormap_var.get())
        fig.colorbar(im)
        
        canvas = FigureCanvasTkAgg(fig, master=self.image_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = canvas
    
    def update_colormap(self, event=None):
        if hasattr(self, 'image_window') and self.image_window.winfo_exists():
            self.view_predicted_image()
    
    def load_excel(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
            
            if file_path:
                # Reset GUI
                self.reset_gui()
                
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                else:
                    self.df = pd.read_excel(file_path)
                
                # Store input filename
                self.input_filename = os.path.splitext(os.path.basename(file_path))[0]
                
                # Identify wavelength columns (numeric column names)
                self.wavelengths = [col for col in self.df.columns 
                                  if str(col).replace('.', '').isdigit() 
                                  and 400 <= float(col) <= 10000]
                
                # Set default min/max wavelengths
                self.min_wave_var.set(min(float(w) for w in self.wavelengths))
                self.max_wave_var.set(max(float(w) for w in self.wavelengths))
                
                # Identify soil property columns
                self.soil_properties = [col for col in self.df.columns 
                                      if not str(col).replace('.', '').isdigit()]
                                      
                # Set default number of components
                for model_type in ["PLS-R", "PCA"]:
                    if model_type in self.model_params:
                        self.model_params[model_type]["params"]["n_components"]["default"] = str(len(self.wavelengths))
                    
                # Update default number of components
                if "n_components" in self.param_vars:
                    self.param_vars["n_components"].set(str(len(self.wavelengths)))
                
                self.status_text.insert(tk.END, f"Loaded file: {self.input_filename}\n")
                self.status_text.insert(tk.END, f"Found {len(self.wavelengths)} wavelengths and {len(self.soil_properties)} properties\n")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.status_text.insert(tk.END, f"Error loading file: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
    
    def check_excel(self):
        try:
            if self.df is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            
            info = f"Dataset Info:\n"
            info += f"Number of samples: {len(self.df)}\n"
            info += f"Wavelength range: {min(self.wavelengths)} - {max(self.wavelengths)} nm\n"
            info += f"Available soil properties: {', '.join(self.soil_properties)}\n"
            
            messagebox.showinfo("Excel File Info", info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check Excel: {str(e)}")
            
    def view_model(self):
        try:
            if self.trained_model is None:
                messagebox.showwarning("Warning", "No model loaded")
                return
            
            model_info = "Model Properties:\n\n"
            model_info += f"Model Type: {self.model_var.get()}\n"
            model_info += f"Preprocessing: {self.preprocess_var.get()}\n"
            model_info += f"Number of Input Features: {len(self.wavelengths) if self.wavelengths else 'Unknown'}\n"
            
            # Add model-specific properties
            if isinstance(self.trained_model, PLSRegression):
                model_info += f"Number of Components: {self.trained_model.n_components}\n"
            elif isinstance(self.trained_model, PCA):
                model_info += f"Number of Components: {self.trained_model.n_components}\n"
            elif isinstance(self.trained_model, RandomForestRegressor):
                model_info += f"Number of Trees: {self.trained_model.n_estimators}\n"
                model_info += f"Max Depth: {self.trained_model.max_depth}\n"
                model_info += f"Min Samples Split: {self.trained_model.min_samples_split}\n"
                model_info += f"Min Samples Leaf: {self.trained_model.min_samples_leaf}\n"
            elif isinstance(self.trained_model, (Ridge, Lasso)):
                model_info += f"Alpha: {self.trained_model.alpha}\n"
            elif isinstance(self.trained_model, SVR):
                model_info += f"Kernel: {self.trained_model.kernel}\n"
                model_info += f"C: {self.trained_model.C}\n"
                model_info += f"Epsilon: {self.trained_model.epsilon}\n"
                if self.trained_model.kernel == 'poly':
                    model_info += f"Degree: {self.trained_model.degree}\n"
            
            messagebox.showinfo("Model Properties", model_info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view model properties: {str(e)}")
            self.status_text.insert(tk.END, f"Error viewing model properties: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
    
    def save_model(self):
        try:
            if self.trained_model is None:
                messagebox.showwarning("Warning", "Please train a model first")
                return
            
            # Generate default filename
            default_filename = generate_model_filename(self.input_filename, self.selected_property, self.model_var.get())
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=[("Joblib files", "*.joblib")],
                initialfile=default_filename
            )
            
            if file_path:
                # Save model and scalers with parameters
                model_data = {
                    'model': self.trained_model,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'wavelengths': self.wavelengths,
                    'filtered_wavelengths': self.filtered_wavelengths,
                    'new_wavelengths': self.new_wavelengths,
                    'model_type': self.model_var.get(),
                    'preprocessing': self.preprocess_var.get(),
                    'model_parameters': {name: var.get() for name, var in self.param_vars.items()},
                    'selected_property': self.selected_property
                }
                joblib.dump(model_data, file_path)
                
                self.status_text.insert(tk.END, f"Model saved to: {file_path}\n")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Joblib files", "*.joblib")])
            
            if file_path:
                # Load model and scalers
                model_data = joblib.load(file_path)
                self.trained_model = model_data['model']
                self.scaler_X = model_data['scaler_X']
                self.scaler_y = model_data['scaler_y']
                self.wavelengths = model_data['wavelengths']
                self.filtered_wavelengths = model_data.get('filtered_wavelengths')
                self.new_wavelengths = model_data.get('new_wavelengths')
                
                # Update GUI to match loaded model
                self.model_var.set(model_data['model_type'])
                self.preprocess_var.set(model_data['preprocessing'])
                
                if 'selected_property' in model_data:
                    self.selected_property = model_data['selected_property']
                
                # Load model parameters if available
                if 'model_parameters' in model_data:
                    self.on_model_change(None)  # Create parameter widgets first
                    for param_name, param_value in model_data['model_parameters'].items():
                        if param_name in self.param_vars:
                            self.param_vars[param_name].set(param_value)
                
                self.status_text.insert(tk.END, f"Model loaded from: {file_path}\n")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Raster files", "*.tif *.tiff")])
            
            if file_path:
                with rasterio.open(file_path) as src:
                    self.image_data = src.read()
                    self.image_meta = src.meta.copy()
                    
                    self.status_text.insert(tk.END, f"Loaded image: {file_path}\n")
                    self.status_text.insert(tk.END, f"Image shape: {self.image_data.shape}\n")
                    self.status_text.insert(tk.END, "="*50 + "\n")
                    self.status_text.see(tk.END)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def save_prediction(self):
        try:
            if not hasattr(self, 'image_data') or self.image_data is None:
                messagebox.showwarning("Warning", "Please load an image first")
                return
            
            if self.trained_model is None:
                messagebox.showwarning("Warning", "Please load a model first")
                return
            
            # Check if number of bands matches wavelengths
            if self.image_data.shape[0] != len(self.wavelengths):
                messagebox.showerror("Error", 
                    f"Image has {self.image_data.shape[0]} bands but model expects {len(self.wavelengths)} bands")
                return
            
            self.update_progress(0, "Processing image for prediction...")
            
            # Process image with same preprocessing as training
            processed_data, original_shape = process_image_for_prediction(
                self.image_data, self.wavelengths, self.preprocess_var.get(),
                self.scaler_X, self.filtered_wavelengths, self.new_wavelengths
            )
            
            self.update_progress(40, "Making predictions...")
            
            # Make prediction
            predictions_scaled = self.trained_model.predict(processed_data)
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
            
            self.update_progress(80, "Saving results...")
            
            # Save prediction
            file_path = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("GeoTIFF files", "*.tif")])
            
            if file_path:
                self.predicted_image = save_prediction_image(predictions, original_shape, self.image_meta, file_path)
                
                self.update_progress(100, f"Prediction saved to: {file_path}")
                self.status_text.insert(tk.END, "="*50 + "\n")
                self.status_text.see(tk.END)
                
                # Enable view image button
                self.view_image_btn.config(state='normal')
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save prediction: {str(e)}")
            self.status_text.insert(tk.END, f"Error saving prediction: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
    
    def select_soil_property(self):
        try:
            if self.soil_properties is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            
            # Check if the dialog already exists and is open
            if hasattr(self, 'selection_window') and self.selection_window.winfo_exists():
                self.selection_window.lift()
                return
            
            self.selection_window = tk.Toplevel(self)
            self.selection_window.title("Select Soil Property")
            
            listbox = tk.Listbox(self.selection_window, selectmode=tk.SINGLE)
            for prop in self.soil_properties:
                listbox.insert(tk.END, prop)
            listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            def on_select():
                if listbox.curselection():
                    self.selected_property = self.soil_properties[listbox.curselection()[0]]
                    self.status_text.insert(tk.END, f"Selected property: {self.selected_property}\n")
                    self.status_text.insert(tk.END, "="*50 + "\n")
                    self.status_text.see(tk.END)
                    self.selection_window.destroy()
            
            ttk.Button(self.selection_window, text="Select", command=on_select).pack(pady=5)

            def on_close():
                self.selection_window.destroy()
                del self.selection_window
            
            self.selection_window.protocol("WM_DELETE_WINDOW", on_close)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select property: {str(e)}")
    
    def start_analysis(self):
        try:
            if not hasattr(self, 'selected_property') or self.selected_property is None:
                messagebox.showwarning("Warning", "Please select a soil property first")
                return
            
            if self.df is None:
                messagebox.showwarning("Warning", "Please load an Excel file first")
                return
            
            # Validate test size
            try:
                test_size = float(self.test_size_var.get())
                if not 0 < test_size < 1:
                    messagebox.showerror("Error", "Test size must be between 0 and 1")
                    return
            except ValueError:
                messagebox.showerror("Error", "Invalid test size value")
                return
            
            n_cores = int(self.cores_var.get())
            self.status_text.insert(tk.END, f"Using {n_cores} cores for model training\n")
            
            self.update_progress(0, "Starting analysis...")
            
            # Filter wavelengths based on spectral configuration
            try:
                self.filtered_wavelengths, self.new_wavelengths = filter_wavelengths(
                    self.wavelengths, self.min_wave_var.get(), self.max_wave_var.get(),
                    self.resampling_var.get(), self.spacing_var.get()
                )
            except Exception as e:
                messagebox.showerror("Wavelength Filtering Error", str(e))
                return
            
            # Prepare data
            X = self.df[self.wavelengths].values
            y = self.df[self.selected_property].values
            
            # Apply wavelength filtering
            wavelength_indices = [i for i, w in enumerate(self.wavelengths) 
                                if float(w) >= min(self.filtered_wavelengths) 
                                and float(w) <= max(self.filtered_wavelengths)]
            X = X[:, wavelength_indices]
            
            # Apply resampling if needed
            if self.new_wavelengths is not None:
                try:
                    X = safe_interpolate_spectra(X, self.filtered_wavelengths, self.new_wavelengths)
                    self.filtered_wavelengths = self.new_wavelengths
                except Exception as e:
                    messagebox.showerror("Interpolation Error", str(e))
                    return
            
            self.update_progress(10, "Calculating statistics...")
            
            # Calculate data statistics
            data_stats = {
                'Input Data Statistics': calculate_statistics(y),
                'Spectral Statistics': {
                    'Mean Reflectance': calculate_statistics(np.mean(X, axis=1)),
                    'Min Reflectance': calculate_statistics(np.min(X, axis=1)),
                    'Max Reflectance': calculate_statistics(np.max(X, axis=1))
                }
            }
            
            self.update_progress(20, "Preprocessing spectra...")
            
            # Preprocess spectra
            X = preprocess_spectra(X, self.preprocess_var.get())
            
            self.update_progress(30, "Splitting data...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            self.update_progress(40, "Scaling data...")
            
            # Scale data
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            self.update_progress(50, "Training model...")
            
            # Get parameters for the selected model
            model_type = self.model_var.get()
            params = {}
            for param_name, param_var in self.param_vars.items():
                param_value = param_var.get()
                params[param_name] = parse_parameter_value(param_name, param_value, param_name)
            
            # Component optimization for PCA and PLSR
            component_optimization_results = None
            if self.optimize_components_var.get() and model_type in ["PLS-R", "PCA"]:
                max_components = min(50, X_train.shape[1], X_train.shape[0] - 1)
                optimal_components, components, rmse_values, r2_values = optimize_components_parallel(
                    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
                    model_type, max_components, self.scaler_y, n_cores
                )
                
                if optimal_components is not None:
                    params['n_components'] = optimal_components
                    component_optimization_results = {
                        'Components': components,
                        'RMSE': rmse_values,
                        'R2_Score': r2_values,
                        'optimal_components': optimal_components
                    }
                    self.status_text.insert(tk.END, f"Optimal components: {optimal_components}\n")
            
            # Create and train model
            self.trained_model = create_model(model_type, params, n_cores)
            
            # Handle PCA separately
            if model_type == "PCA":
                pca = self.trained_model
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                self.trained_model = LinearRegression()
                self.trained_model.fit(X_train_pca, y_train_scaled)
                y_pred_scaled = self.trained_model.predict(X_test_pca)
                y_train_pred_scaled = self.trained_model.predict(X_train_pca)
            else:
                self.trained_model.fit(X_train_scaled, y_train_scaled)
                y_pred_scaled = self.trained_model.predict(X_test_scaled)
                y_train_pred_scaled = self.trained_model.predict(X_train_scaled)
            
            self.update_progress(70, "Making predictions...")
            
            # Convert predictions back to original scale
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics for both train and test sets
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            
            # Cross-validation if requested
            cv_results = None
            if self.cv_strategy_var.get() != "None":
                self.update_progress(85, "Performing cross-validation...")
                
                # Prepare CV parameters
                cv_params = {}
                if self.cv_strategy_var.get() == "K-Fold":
                    cv_params['k_folds'] = int(self.k_folds_var.get())
                    cv_params['shuffle'] = self.shuffle_var.get().lower() == 'true'
                elif self.cv_strategy_var.get() == "Leave-P-Out":
                    cv_params['p_out'] = int(self.p_out_var.get())
                
                # Perform cross-validation
                cv_rmse_scores, cv_r2_scores, cv_rmse_mean, cv_r2_mean, cv_rmse_std, cv_r2_std = perform_cross_validation(
                    X_train_scaled, y_train_scaled, self.trained_model, 
                    self.cv_strategy_var.get(), cv_params, self.scaler_y, n_cores
                )
                
                cv_results = {
                    'strategy': self.cv_strategy_var.get(),
                    'cv_rmse_scores': cv_rmse_scores,
                    'cv_r2_scores': cv_r2_scores,
                    'rmse_mean': cv_rmse_mean,
                    'rmse_std': cv_rmse_std,
                    'r2_mean': cv_r2_mean,
                    'r2_std': cv_r2_std,
                    'parameters': cv_params
                }
            
            self.update_progress(90, "Calculating correlations...")
            
            # Calculate correlations for correlogram
            correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
            
            # Ask user for output filename
            default_filename = generate_default_filename(self.input_filename, self.selected_property, model_type)
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=default_filename
            )
            
            if not file_path:
                return
            
            self.output_filename = file_path
            
            # Prepare results data
            results_data = {
                'y_test': y_test,
                'y_pred': y_pred,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'test_size': test_size,
                'selected_property': self.selected_property,
                'preprocessing': self.preprocess_var.get(),
                'n_cores': n_cores,
                'model_type': model_type,
                'filtered_wavelengths': self.filtered_wavelengths,
                'correlations': correlations,
                'params': params,
                'cv_results': cv_results,
                'component_optimization_results': component_optimization_results,
                'export_stats': self.export_stats_var.get(),
                'data_stats': data_stats if self.export_stats_var.get() else None
            }
            
            # Save results to Excel
            save_results_to_excel(self.output_filename, results_data)
            
            # Display results
            self.status_text.insert(tk.END, f"Test R² = {test_r2:.3f}, Test RMSE = {test_rmse:.3f}\n")
            self.status_text.insert(tk.END, f"Train R² = {train_r2:.3f}, Train RMSE = {train_rmse:.3f}\n")
            
            if cv_results:
                self.status_text.insert(tk.END, f"CV R² = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}\n")
                self.status_text.insert(tk.END, f"CV RMSE = {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}\n")
            
            self.status_text.insert(tk.END, f"Results saved to {self.output_filename}\n")
            
            self.update_progress(100, f"Analysis complete for {model_type}!")
            self.save_model_btn.config(state='normal')
            self.load_model_btn.config(state='normal')
            self.view_model_btn.config(state='normal')
            
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.status_text.insert(tk.END, f"Error: {str(e)}\n")
            self.status_text.insert(tk.END, "="*50 + "\n")
            self.status_text.see(tk.END)

if __name__ == "__main__":
    app = SpectralAnalyzer()
    app.mainloop()