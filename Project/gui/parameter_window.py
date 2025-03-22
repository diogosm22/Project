# gui/parameter_window.py

import tkinter as tk
from ..config import conf_threshold, scale_factor, margin_value  # Absolute import

def get_parameters():
    def apply_settings():
        global conf_threshold, scale_factor, margin_value
        conf_threshold = conf_threshold.get() / 100
        scale_factor = scale_factor.get() / 100
        margin_value = margin_value.get()
        param_window.destroy()

    def exit_program():
        print("Leaving.")
        param_window.destroy()
        exit()

    # Tkinter window for parameter settings
    param_window = tk.Tk()
    param_window.title("Detection Parameters")
    param_window.geometry("540x360")
    param_window.configure(bg="#f4f4f9")

    # Add your GUI code here (same as in your original code)

    param_window.mainloop()