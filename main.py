import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import tkinter.ttk as ttk

#Get rid of zeros and extreme values in a DataFrame
def remove_outliers_by_percentage(df, maxDeviation):
    means = df.mean()
    allowed_deviation = means * (maxDeviation / 100.0)
    def filter_func(x, mean, allowed):
        if pd.isna(x):
            return np.nan
        if abs(x - mean) > allowed:
            return np.nan
        return x
    for col in df.columns:
        df[col] = df[col].apply(lambda x: filter_func(x, means[col], allowed_deviation[col]))
    return df

def gui_main():
    # Browse for file
    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            entry_file.delete(0, tk.END)
            entry_file.insert(0, file_path)

    # clean the file by removing zeros and extreme values if the user wants to and return the cleaned DataFrame
    def process_file():
        file_path = entry_file.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a CSV file.") # check for empty file path
            return
        #Let the user choose if they want to delete zeros and extreme values
        delete_zeros = var_delete_zeros.get()
        delete_extremes = var_delete_extremes.get()
        try:
            df = pd.read_csv(file_path, header=None, dtype=str)
            def try_float(x):
                try:
                    val = float(x)
                    if delete_zeros and val == 0.0: #if delete_zeros is true Zeros get deleted
                        return np.nan
                    return val
                except Exception:
                    return np.nan
            df = df.applymap(try_float)
            df_clean = df.dropna().reset_index(drop=True)
            if delete_extremes: # if delete_extremes is true extreme values get deleted
                try:
                    max_dev = float(entry_max_deviation.get())
                except Exception:
                    messagebox.showerror("Error", "Please enter a valid max deviation (percent).") # check if deviation has a value between 0 and 1
                    return
                df_clean = remove_outliers_by_percentage(df_clean, max_dev)
                df_clean = df_clean.dropna().reset_index(drop=True)
            gui_main.df_clean = df_clean
            update_table(df_clean)
            def calculate_average():
                if not hasattr(gui_main, 'df_clean'):
                    messagebox.showerror("Error", "Please process a file first.")
                    return
                mean_val = gui_main.df_clean.iloc[:, 0].mean()
                update_table(gui_main.df_clean, show_mean=True, mean_val=mean_val)
                messagebox.showinfo("Column Mean", f"Mean: {mean_val:.3f}")
                return mean_val
            calculate_average()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{e}")

    root = tk.Tk()
    root.title("PAP 1 Means and Deviations Calculator (FEZ)")

    # File selection
    frame_file = tk.Frame(root)
    frame_file.pack(padx=10, pady=2, fill='x')
    tk.Label(frame_file, text="CSV File:").pack(side='left')
    entry_file = tk.Entry(frame_file, width=40)
    entry_file.pack(side='left', padx=5)
    entry_file.insert(0, "example_data/example1.csv")  # Set default CSV path
    tk.Button(frame_file, text="Browse", command=select_file).pack(side='left', padx=5)

    # Cleaning options
    frame_options = tk.Frame(root)
    frame_options.pack(padx=10, pady=2, fill='x')
    var_delete_zeros = tk.BooleanVar(value=True)
    chk_delete_zeros = tk.Checkbutton(frame_options, text="Delete zeros", variable=var_delete_zeros)
    chk_delete_zeros.pack(side='left', padx=5)
    var_delete_extremes = tk.BooleanVar(value=False)
    chk_delete_extremes = tk.Checkbutton(frame_options, text="Delete extremes", variable=var_delete_extremes)
    chk_delete_extremes.pack(side='left', padx=5)
    tk.Label(frame_options, text="Max Deviation (%):").pack(side='left', padx=5)
    entry_max_deviation = tk.Entry(frame_options, width=5)
    entry_max_deviation.insert(0, "10")
    entry_max_deviation.pack(side='left', padx=5)

    # Process button
    frame_process = tk.Frame(root)
    frame_process.pack(padx=10, pady=2, fill='x')
    btn_process = tk.Button(frame_process, text="Process Data", command=process_file)
    btn_process.pack(side='left', padx=5)

    # Calculation buttons
    frame_calc = tk.Frame(root)
    frame_calc.pack(padx=10, pady=2, fill='x')

    # Results row (for mean and other single values)
    frame_results = tk.Frame(root)
    frame_results.pack(padx=10, pady=2, fill='x')
    label_mean = tk.Label(frame_results, text="Mean: -")
    label_mean.pack(side='left', padx=5)

    # Data table
    frame_table = tk.Frame(root)
    frame_table.pack(padx=10, pady=10, fill='both', expand=True)
    tree = ttk.Treeview(frame_table, show='headings')
    tree.pack(side='left', fill='both', expand=True)
    scrollbar = tk.Scrollbar(frame_table, orient='vertical', command=tree.yview)
    scrollbar.pack(side='right', fill='y')
    tree.configure(yscrollcommand=scrollbar.set)

    def update_table(df, show_mean=False, mean_val=None):
        tree.delete(*tree.get_children())
        tree['columns'] = ['Index', 'Value', 'ADfM']
        tree.heading('Index', text='Index')
        tree.heading('Value', text='Value')
        tree.heading('ADfM', text='Average Deviation from Mean')
        tree.column('Index', width=5, anchor='center')
        tree.column('Value', width=60, anchor='center')
        tree.column('ADfM', width=60, anchor='center')
        means = df.iloc[:, 0].mean()
        deviations = (df.iloc[:, 0] - means).abs()
        for idx, val in df.iloc[:, 0].items():
            dev = deviations[idx] if pd.notna(val) else ''
            tree.insert('', 'end', values=[idx, f'{val:.3f}' if pd.notna(val) else '', f'{dev:.3f}' if pd.notna(dev) else ''])
        # Update mean label above table if requested
        if show_mean and mean_val is not None:
            label_mean.config(text=f"Mean: {mean_val:.3f}")
        else:
            label_mean.config(text="Mean: -")

    root.mainloop()

gui_main()