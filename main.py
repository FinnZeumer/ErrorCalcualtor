import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from math import erf
import matplotlib.pyplot as plt

versuchsNummern = [00, 11, 12, 13, 14, 15, 21, 22, 23, 25, 26, 31, 33, 34, 35, 41, 42]
versuchsNummernStr = [str(num) for num in versuchsNummern]

# Get rid of zeros and extreme values in a DataFrame
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

            # Anzahl der Messwerte
            def samples_count():
                if not hasattr(gui_main, 'df_clean'):
                    messagebox.showerror("Error", "Please process a file first.")
                    return
                count = gui_main.df_clean.shape[0]
                label_sampels.config(text=f"Anzahl Messwerte: {count}")
                return count
            samples_count()
            # Bestimmung des arithmetischen Mittels
            def calculate_average():
                if not hasattr(gui_main, 'df_clean'):
                    messagebox.showerror("Error", "Please process a file first.")
                    return
                mean_val = gui_main.df_clean.iloc[:, 0].mean()
                update_table(gui_main.df_clean, show_mean=True, mean_val=mean_val)
                # messagebox.showinfo("arithmetisches Mittel der Einzelmessunge", f"arithmetischer Mittelwert: {mean_val:.3f}")
                return mean_val
            calculate_average()

            # Bestimmung der Standardabweichung
            def calculate_standard_deviation():
                if not hasattr(gui_main, 'df_clean'):
                    messagebox.showerror("Error", "Please process a file first.")
                    return
                std_dev = gui_main.df_clean.iloc[:, 0].std()
                update_table(gui_main.df_clean, show_std_dev=True, std_dev=std_dev)
                # messagebox.showinfo("Standardabweichung", f"Standardabweichung: {std_dev:.3f}")
                return std_dev
            calculate_standard_deviation()

            # Bestimmung der Variation
            def calculate_variation():
                if not hasattr(gui_main, 'df_clean'):
                    messagebox.showerror("Error", "Please process a file first.")
                    return
                variation = gui_main.df_clean.iloc[:, 0].var()
                label_variation.config(text=f"Variation \u03C3² : {variation:.3f}")
                # messagebox.showinfo("Variation", f"Variation: {variation:.3f}")
                return variation
            calculate_variation()

            # Bestimmung des statistischen Fehlers
            def calculate_statistical_error():
                if not hasattr(gui_main, 'df_clean'):
                    messagebox.showerror("Error", "Please process a file first.")
                    return
                count = samples_count()
                if count < 2:
                    messagebox.showerror("Error", "Not enough data to calculate statistical error.")
                    return
                std_dev = gui_main.df_clean.iloc[:, 0].std()
                stat_error = std_dev / (count ** 0.5)
                label_stat_error.config(text=f"Statistischer Fehler: \u00B1{stat_error:.3f}")
                # messagebox.showinfo("Statistischer Fehler", f"Statistischer Fehler: {stat_error:.3f}")
                return stat_error
            calculate_statistical_error()            

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{e}")

    root = tk.Tk()
    root.title("PAP 1 - Fehlerrechner (FEZ)")

    style = ttk.Style()
    root.configure(background="white")
    style.theme_use("clam")  # Modern ttk theme
    style.configure("TFrame", background="white")
    style.configure("TLabelframe", background="white", borderwidth=2, relief="groove")
    style.configure("TLabelframe.Label", font=("Arial", 12, "bold"),background="white", foreground="#C61826")
    style.configure("TLabel", background="white", font=("Arial", 10))
    style.configure("TButton", font=("Arial", 10, "bold"), padding=5, background="#F4F1EA", foreground="#590D08")
    style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
    style.configure("Treeview", rowheight=24)
    style.configure("TCheckbutton", background="white", font=("Arial", 10))

# Erstellung eines Plots mit den ausgewählten Optionen
    def plot_data():
        plt.close('all')
        fig, ax = plt.subplots()

        data = gui_main.df_clean.iloc[:, 0]
        bins = int(entry_accuracy.get())
        title = entry_title.get()
        xlabel = entry_x_label.get()
        ylabel = entry_y_label.get()

        # Histogram
        if var_show_histogram.get():
            ax.hist(data, bins=bins, edgecolor='black', alpha=0.6, label='Histogramm')

        # Normalized Histogram
        if var_show_normal_histogram.get():
            ax.hist(data, bins=bins, density=True, color='#F4F1EA', edgecolor='black', alpha=0.6, label='Normiertes Histogramm')

        # Gaussian fit
        if var_schow_gaussian_fit.get():
            mu, sigma = data.mean(), data.std()
            count, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)
            ax.plot(bin_centers, gaussian, '-', color='#C61826', label=f'Gauß\'sche Normalverteilung')

        # Mean line
        if var_show_mean.get():
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mittelwert: {mean_val:.3f}')

        # Standard deviation lines
        if var_show_std_dev.get():
            std_dev = data.std()
            mean_val = data.mean()  # ensure mean is defined
            ax.axvline(mean_val + std_dev, color='green', linestyle='dotted', linewidth=1, label=f'Stdandardabweichung: {std_dev:.3f}')
            ax.axvline(mean_val - std_dev, color='green', linestyle='dotted', linewidth=1)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.canvas.manager.set_window_title(title)

        plt.show()

    def wahrscheinlichkeits_integral_berechnen():
        try:
            data = gui_main.df_clean.iloc[:, 0]
            mu = data.mean()
            sigma = data.std()

            # Get input values from GUI
            a = float(wahrsch_von.get())
            b = float(wahrsch_bis.get())

            if a >= b:
                messagebox.showerror("Error", "The 'from' value must be less than the 'to' value.")
                return

            # Gaussian probability using erf from math module
            prob_val = 0.5 * (erf((b - mu) / (sigma * np.sqrt(2))) - erf((a - mu) / (sigma * np.sqrt(2)))) *100

            # messagebox.showinfo("Wahrscheinlichkeit", f"Wahrscheinlichkeit G({a},{b}) = {prob_val:.5f}")
            prob_var.set(f"Wahrscheinlichkeit: {prob_val:.3f} %")
            return prob_val
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate probability:\n{e}")


    # GUI Layout
    frame_header = ttk.Frame(root, padding=10)
    frame_header.pack(padx=10, pady=5, fill="x")

    label_title = ttk.Label(frame_header, text="Fehlerrechner (by FEZ)", font=("Arial", 16, "bold"), foreground="#C61826") 
    label_title.pack(side="top")
    label_subtitle = ttk.Label(frame_header, text="PAP 1 - Praktische Arbeit in der Physik", font=("Arial", 10), foreground="#590D08")
    label_subtitle.pack(side="top", pady=2)

    # basic selection row with logo
    frame_basic = ttk.Frame(root, padding=10)
    frame_basic.pack(padx=10, pady=5, fill="x")

    label_versuch = ttk.Label(frame_basic, text="Versuch: ", font=("Arial", 12, "bold"), foreground="#590D08")
    label_versuch.pack(side="left")
    versuchsNummerSelection = tk.IntVar(value="11")
    entry_versuch = ttk.OptionMenu(frame_basic, versuchsNummerSelection, *versuchsNummern)
    entry_versuch.pack(side="left", padx=5)

    # include image 
    img = Image.open("Logo_Fabe_Tranzparent_Sicherheitslücken.png")
    img = img.resize((190, 100))
    img_tk = ImageTk.PhotoImage(img)
    label_img = ttk.Label(frame_basic, image=img_tk)
    label_img.image = img_tk  # keep a reference
    label_img.pack(side="right")


    # File selection
    frame_file = ttk.LabelFrame(root, text="CSV-Datei auswählen", padding=10)
    frame_file.pack(padx=10, pady=5, fill="x")
    ttk.Label(frame_file, text="CSV File:").pack(side="left")
    entry_file = ttk.Entry(frame_file, width=40)
    entry_file.pack(side="left", padx=5)
    entry_file.insert(0, "example_data/example1.csv")
    ttk.Button(frame_file, text="Browse", command=select_file).pack(side="left", padx=5)

    # Cleaning options
    frame_options = ttk.LabelFrame(root, text="Datenbereinigung", padding=10)
    frame_options.pack(padx=10, pady=5, fill="x")

    var_delete_zeros = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame_options, text="Null Zeilen löschen", variable=var_delete_zeros).pack(side="left", padx=5)

    var_delete_extremes = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame_options, text="Extreme Werte löschen", variable=var_delete_extremes).pack(side="left", padx=5)

    ttk.Label(frame_options, text="Max. relative Abweichung (%):").pack(side="left", padx=5)
    entry_max_deviation = ttk.Entry(frame_options, width=5)
    entry_max_deviation.insert(0, "10")
    entry_max_deviation.pack(side="left", padx=5)

    # Process button
    ttk.Button(frame_options, text="Daten auswerten", command=process_file).pack(side="right", padx=5)

    # Plot options
    frame_plot_options = ttk.LabelFrame(root, text="Graphische Darstellung", padding=10)
    frame_plot_options.pack(padx=10, pady=5, fill="x")

    var_show_histogram = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame_plot_options, text="Histogramm anzeigen", variable=var_show_histogram).pack(side="left", padx=5)

    var_show_normal_histogram = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame_plot_options, text="Normiertes Histogramm anzeigen", variable=var_show_normal_histogram).pack(side="left", padx=5)

    var_schow_gaussian_fit = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame_plot_options, text="Gaußverteilung anzeigen", variable=var_schow_gaussian_fit).pack(side="left", padx=5)

    var_show_mean = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame_plot_options, text="Arithmetisches Mittel anzeigen", variable=var_show_mean).pack(side="left", padx=5)

    var_show_std_dev = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame_plot_options, text="Standardabweichung anzeigen", variable=var_show_std_dev).pack(side="left", padx=5)

    ttk.Label(frame_plot_options, text="Anzahl Balken:").pack(side="left", padx=5)
    entry_accuracy = ttk.Entry(frame_plot_options, width=5)
    entry_accuracy.insert(0, "30")
    entry_accuracy.pack(side="left", padx=5)

    ttk.Label(frame_plot_options, text="Titel:").pack(side="left", padx=5)
    entry_title = ttk.Entry(frame_plot_options, width=35)
    entry_title.insert(0, "Histogramm der Messwerte")
    entry_title.pack(side="left", padx=5)

    ttk.Label(frame_plot_options, text="x-Achsen Label:").pack(side="left", padx=5)
    entry_x_label = ttk.Entry(frame_plot_options, width=35)
    entry_x_label.insert(0, "Messwerte")
    entry_x_label.pack(side="left", padx=5)

    ttk.Label(frame_plot_options, text="y-Achsen Label:").pack(side="left", padx=5)
    entry_y_label = ttk.Entry(frame_plot_options, width=35)
    entry_y_label.insert(0, "Anzahl/Warscheinlichkeitsdichte")
    entry_y_label.pack(side="left", padx=5)

    ttk.Button(frame_plot_options, text="Plot", command=plot_data).pack(side="right", padx=5)


    # Results
    frame_results = ttk.LabelFrame(root, text="Ergebnisse", padding=10)
    frame_results.pack(padx=10, pady=5, fill="x")

    label_mean = ttk.Label(frame_results, text="Arithmetischer Mittelwert: -")
    label_mean.pack(side="left", padx=5)

    label_std_dev = ttk.Label(frame_results, text="Absolute Standardabweichung \u03C3 : -")
    label_std_dev.pack(side="left", padx=5)

    label_variation = ttk.Label(frame_results, text="Variation \u03C3² : -")
    label_variation.pack(side="left", padx=5)

    label_stat_error = ttk.Label(frame_results, text="Statistischer Fehler: -")
    label_stat_error.pack(side="left", padx=5)

    label_sampels = ttk.Label(frame_results, text="Anzahl Messwerte: -")
    label_sampels.pack(side="left", padx=5)

    # Kalkulationen
    frame_calculations = ttk.LabelFrame(root, text="Kalkulationen", padding=10)
    frame_calculations.pack(padx=10, pady=5, fill="x")

    ttk.Label(frame_calculations, text="Wahrscheinlickeit G(a,b), dass ein Event im Intervall [a,b] eintifft.").pack(side="left", padx=5)
    ttk.Label(frame_calculations, text="von").pack(side="left", padx=5)
    wahrsch_von = ttk.Entry(frame_calculations, width=5)
    wahrsch_von.insert(0, "-1")
    wahrsch_von.pack(side="left", padx=5)
    ttk.Label(frame_calculations, text="bis").pack(side="left", padx=5)
    wahrsch_bis = ttk.Entry(frame_calculations, width=5)
    wahrsch_bis.insert(0, "1")
    wahrsch_bis.pack(side="left", padx=5)
    ttk.Button(frame_calculations, text="Wahrscheinlichkeit berechnen", command=wahrscheinlichkeits_integral_berechnen).pack(side="left", padx=5)

    prob_var = tk.StringVar(value="Wahrscheinlichkeit: -")
    prob_label = ttk.Label(frame_calculations, textvariable=prob_var)
    prob_label.pack(side="left", padx=5)

    # Data table
    frame_table = ttk.LabelFrame(root, text="Tabelle der Einzelwerte", padding=10)
    frame_table.pack(padx=10, pady=10, fill="both", expand=True)

    tree = ttk.Treeview(frame_table, show="headings")
    tree.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(frame_table, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)

    # Export settings
    frame_results = ttk.LabelFrame(root, text="Export Einstellungen", padding=10)
    frame_results.pack(padx=10, pady=5, fill="x")


    # Folder selection
    ttk.Label(frame_results, text="Export Ordner:").pack(side="left", padx=5)
    entry_export_folder = ttk.Entry(frame_results, width=40)
    entry_export_folder.pack(side="left", padx=5)
    entry_export_folder.insert(0, "C:/Folders/Uni-Heidelberg/Praktika/PAP1/PythonFehlerrechnung/example_data/")
    ttk.Button(
        frame_results,
        text="Browse",
        command=lambda: entry_export_folder.delete(0, tk.END) or entry_export_folder.insert(0, filedialog.askdirectory())
    ).pack(side="left", padx=5)

    # Filename
    ttk.Label(frame_results, text="Dateiname (ohne Endung):").pack(side="left", padx=5)
    entry_export_filename = ttk.Entry(frame_results, width=20)
    entry_export_filename.pack(side="left", padx=5)
    entry_export_filename.insert(0, "output")  # default filename

    # File type
    ttk.Label(frame_results, text="Dateityp:").pack(side="left", padx=5)
    export_file_type = tk.StringVar(value="tex")
    ttk.OptionMenu(frame_results, export_file_type, "tex", "csv").pack(side="left", padx=5)

    # Export button
    ttk.Button(
        frame_results,
        text="Tabelle exportieren",
        command=lambda: export_table()
    ).pack(side="right", padx=5)

    def export_table():
        if not hasattr(gui_main, 'df_clean'):
            messagebox.showerror("Error", "Please process a file first.")
            return

        folder = entry_export_folder.get()
        filename = entry_export_filename.get()
        file_type = export_file_type.get()

        extension = ".tex" if file_type == "tex" else ".csv"
        export_path = f"{folder}/{filename}{extension}"

        try:
            # Prepare data
            df_export = gui_main.df_clean.copy()
            mean_val = df_export.iloc[:, 0].mean()
            absolut_deviations = (df_export.iloc[:, 0] - mean_val).abs()
            relative_deviations = (absolut_deviations / mean_val * 100).round(3)
            std_dev = df_export.iloc[:, 0].std()
            count = df_export.shape[0]

            df_export['AbsDev'] = absolut_deviations
            df_export['RelDev'] = relative_deviations

            if file_type == "csv":
                # Export as CSV
                if not export_path.lower().endswith(".csv"):
                    export_path += ".csv"
                df_export.to_csv(
                    export_path,
                    index=False,
                    header=['Messwerte', 'Absolute Abweichung', 'Relative Abweichung (\\%)']
                )

            elif file_type == "tex":
                # Export as LaTeX
                if not export_path.lower().endswith(".tex"):
                    export_path += ".tex"

                headers = ['Messwerte', 'Absolute Abweichung', 'Relative Abweichung (%)']
                caption = "Messwerte mit Abweichungen"
                label = "tab:messwerte"

                # Begin LaTeX table
                latex = "\\begin{table}[h!]\n\\centering\n"
                latex += f"\\caption{{{caption}}}\n"
                latex += f"\\label{{{label}}}\n"
                latex += "\\begin{tabular}{" + " | ".join(["l"] + ["r"] * (len(headers)-1)) + "}\n"
                latex += " \\hline\n"
                latex += " & ".join(headers) + " \\\\\n"
                latex += " \\hline\n"

                for row in df_export.values:
                    latex += " & ".join(map(str, row)) + " \\\\\n"

                latex += " \\hline\n\\end{tabular}\n\\end{table}\n\n"

                # Add summary statistics
                latex += f"Arithmetischer Mittelwert: {mean_val:.3f} \\\\\n"
                latex += f"Standardabweichung $\\sigma$ : $\\pm${std_dev:.3f} \\\\\n"
                latex += f"Statistischer Fehler: $\\pm${(std_dev / (count ** 0.5)):.3f} \\\\\n"
                latex += f"Anzahl Messwerte: {count} \n"

                with open(export_path, "w") as f:
                    f.write(latex)

            else:
                messagebox.showerror("Error", "Unsupported export file type.")
                return

            messagebox.showinfo("Success", f"Table exported successfully to {export_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export table:\n{e}")

    
    def update_table(df, show_mean=False, mean_val=None, show_std_dev=False, std_dev=None, prob_val=None):
        tree.delete(*tree.get_children())
        tree['columns'] = ['Index', 'Value', 'AbsDev', 'RelDev']
        tree.heading('Index', text='Index')
        tree.heading('Value', text='Messwerte')
        tree.heading('AbsDev', text='Absolute Abweichung vom Mittelwert')
        tree.heading('RelDev', text='Relative Abweichung vom Mittelwert in %')
        tree.column('Index', width=5, anchor='center')
        tree.column('Value', width=60, anchor='center')
        tree.column('AbsDev', width=60, anchor='center')
        tree.column('RelDev', width=60, anchor='center')
        means = df.iloc[:, 0].mean()
        absolut_deviations = (df.iloc[:, 0] - means).abs()
        relative_deviations = (absolut_deviations / means * 100).round(3)
        for row_idx, (idx, val) in enumerate(df.iloc[:, 0].items(), start=1):
            abs_dev = absolut_deviations[idx] if pd.notna(val) else ''
            rel_dev = relative_deviations[idx] if pd.notna(val) else ''
            tree.insert(
                '', 'end',
                values=[
                    row_idx,  # start at 1
                    f'{val:.3f}' if pd.notna(val) else '',
                    f'{abs_dev:.3f}' if pd.notna(abs_dev) else '',
                    f'{rel_dev:.3f}' if pd.notna(rel_dev) else ''
                ]
            )
            # Update mean label above table if requested
        if show_mean and mean_val is not None:
            label_mean.config(text=f"Arithmetischer Mittelwert: {mean_val:.3f}")

        if show_std_dev and std_dev is not None:
            label_std_dev.config(text=f"Absolute Standartabweichung \u03C3 : \u00B1{std_dev:.3f}")

    root.mainloop()

gui_main()