# Level 6 - fully fledged accidental protect

import cv2
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import time
from datetime import datetime, timedelta
import datetime as dt
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
#from io import BytesIO
#import plotly.graph_objects as go
import pandas as pd
import datetime as dt
import tkinter as tk
from tkinter import ttk
import random
import tkinter as tk
import pandas as pd
import serial
import threading
import plc


FLIP_FEED = True

global total_inspected
total_inspected = 0  # Initialize with 0
global cap
global first_pass_right
first_pass_right = 0  # Initialize to 0

# Global constants and variables
THRESHOLD_CSV = 'thresholds.csv'
ROI_CSV = 'roi_coordinates.csv'
LOG_CSV = 'inspection_log.csv'
ROI_FOLDERS = {'ROI1': 'ROI1_Images', 'ROI2': 'ROI2_Images'}

# Credentials
admin_mode = False
PASSWORD = "1"  # Replace with your desired password

thresholds = {}
roi_coordinates = {}
reference_roi_coordinates = {}
cap = None
trigger_count = 0
ok_count = 0
roi_status = {}
roi_color = {}

def load_thresholds():
    """Load thresholds, zoom, and focus from the CSV file."""
    if os.path.exists(THRESHOLD_CSV):
        df = pd.read_csv(THRESHOLD_CSV, index_col='ROI')
        if 'Zoom' not in df.columns:
            df['Zoom'] = 1.0  # Default zoom value
        if 'Focus' not in df.columns:
            df['Focus'] = 0  # Default focus value
        thresholds_loaded = df[['Threshold', 'Display Name', 'Zoom', 'Focus']].to_dict('index')
    else:
        thresholds_loaded = {}

    # Initialize missing keys for all ROIs
    for roi in roi_coordinates.keys(): #['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5']:
        if roi not in thresholds_loaded:
            thresholds_loaded[roi] = {'Threshold': 0.7, 'Display Name': roi, 'Zoom': 1.0, 'Focus': 0}

    return thresholds_loaded


def load_roi_coordinates():
    if os.path.exists(ROI_CSV):
        df = pd.read_csv(ROI_CSV, index_col='ROI')
        # Ensure 'Display Name' is set if missing
        if 'Display Name' not in df.columns:
            df['Display Name'] = df.index
        coords_dict = df[['x1', 'y1', 'x2', 'y2', 'Display Name']].to_dict('index')
        # Keep all ROIs, but fix zero-size ones
        for roi, coords in coords_dict.items():
            if coords['x1'] == coords['x2'] or coords['y1'] == coords['y2']:
                print(f"Skipping ROI {roi} due to zero size")
        return coords_dict
    return {}

thresholds = load_thresholds()
roi_coordinates = load_roi_coordinates()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# Make sure every ROI has required structure even if CSV is empty
for roi in roi_coordinates.keys():
    if 'Display Name' not in roi_coordinates[roi]:
        roi_coordinates[roi]['Display Name'] = roi
        
ROI_FOLDERS = {roi: os.path.join(BASE_PATH, f"{roi}_Images") for roi in roi_coordinates}
# Ensure directories exist at absolute paths
for folder in ROI_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)
roi_status = {roi: "Waiting" for roi in roi_coordinates.keys()}
roi_color = {roi: (255, 0, 0) for roi in roi_coordinates.keys()}  # Blue before trigger

def save_thresholds():
    """Save thresholds, zoom, and focus values to the CSV file."""
    active_rois = list(roi_coordinates.keys())
    filtered_thresholds = {roi: thresholds[roi] for roi in active_rois if roi in thresholds}
    df = pd.DataFrame.from_dict(filtered_thresholds, orient='index')
    df.to_csv(THRESHOLD_CSV, index_label='ROI')

# UI function for saving updated thresholds from entries
def save_thresholds_ui(self):
    """Save threshold, zoom, and focus values to the thresholds dictionary and CSV."""
    for roi, entry in self.threshold_entries.items():
        thresholds[roi]['Threshold'] = float(entry.get())
        thresholds[roi]['Zoom'] = self.zoom_sliders[roi].get()
        thresholds[roi]['Focus'] = self.focus_sliders[roi].get()

    save_thresholds()
    messagebox.showinfo("Success", "Thresholds, Zoom, and Focus saved successfully.")

def save_roi_coordinates():
    """Save only active ROI coordinates and display names to the CSV file."""
    # Get the active ROI names based on the current number of ROIs
    active_rois = list(roi_coordinates.keys())
    
    # Filter roi_coordinates to include only active ROIs
    filtered_roi_coordinates = {roi: roi_coordinates[roi] for roi in active_rois if roi in roi_coordinates}
    
    # Convert the filtered coordinates to a DataFrame and save to CSV
    df = pd.DataFrame.from_dict(filtered_roi_coordinates, orient='index')
    df.to_csv(ROI_CSV, index_label='ROI')
    print(f"ROI Coordinates saved to CSV: {filtered_roi_coordinates}")

def update_log(timestamp, roi_statuses):
    log_line = f"{timestamp}," + ",".join([f"{roi},{status}" for roi, status in roi_statuses.items()]) + "\n"
    with open(LOG_CSV, 'a') as file:
        file.write(log_line)
        
def create_efficiency_gauge(canvas, efficiency):
    """Draw a simple efficiency gauge on the given canvas, filling from left to right."""
    canvas.delete("all")  # Clear previous drawings
    
    # Draw the gauge background (gray arc from left to right)
    canvas.create_arc(10, 10, 190, 190, start=180, extent=180, fill="#ddd", outline="")
    
    # Draw the efficiency level (green arc filling from left to right)
    angle = int(180 * efficiency / 100)
    canvas.create_arc(10, 10, 190, 190, start=180, extent=-angle, fill="#4CAF50", outline="")
    
    # Draw the center circle for aesthetic
    canvas.create_oval(80, 80, 120, 120, fill="white", outline="")
    
    # Add the percentage text in the center
    #canvas.create_text(100, 100, text=f"{efficiency:.0f}%", font=("Arial", 16, "bold"), fill="black")

def display_summary_table(parent):
    """Display a styled summary table with black text and increased height."""
    # Data to display in the table
    data = {
        "Shift": ["Shift A", "Shift B", "Shift C", "Overall"],
        "Efficiency": ["85.00%", "90.50%", "78.30%", "88.27%"],
        "Total Inspected": [25, 30, 20, 75],
        "OK": [22, 28, 15, 65],
        "NOT OK": [3, 2, 5, 10],
    }

    # Create a DataFrame for easy manipulation
    summary_df = pd.DataFrame(data)

    # Create the Treeview widget for the table
    columns = list(summary_df.columns)
    tree = ttk.Treeview(
        parent,
        columns=columns,
        show="headings",
        height=len(summary_df) + 0,  # Increased height by approx. 35%
    )
    tree.grid(row=0, column=0, padx=0, pady=0)

    # Style configuration
    style = ttk.Style()
    style.configure(
        "Treeview",
        rowheight=40,  # Increased row height for better visibility
        font=("Arial", 14),  # Font for table rows
        background="#ffffff",  # White background for rows
        foreground="#000000",  # Black text color
        fieldbackground="#ffffff",  # White background for fields
    )
    style.configure(
        "Treeview.Heading",
        font=("Arial", 16, "bold"),
        foreground="#000000",  # Black text for headers
        background="#e0e0e0",  # Light gray background for headers
    )
    style.map("Treeview", background=[("selected", "#cce5ff")])  # Highlight color on selection

    # Add column headers
    for col in columns:
        tree.heading(col, text=col, anchor="center")
        tree.column(col, anchor="center", width=200)

    # Insert data into the table
    for _, row in summary_df.iterrows():
        tree.insert("", "end", values=row.tolist())

    # Add grid lines using alternating row colors (manual workaround for borders)
    for index, item in enumerate(tree.get_children()):
        if index % 2 == 0:
            tree.item(item, tags=("even",))
        else:
            tree.item(item, tags=("odd",))

    tree.tag_configure("even", background="#f9f9f9", foreground="#000000")
    tree.tag_configure("odd", background="#ffffff", foreground="#000000")

    # Add an outline for the table
    tree.grid_configure(padx=0, pady=0)  # Ensure padding around the table

def get_shift_data():
    # Load data from CSV
    if not os.path.exists(LOG_CSV):
        return pd.DataFrame(columns=["Timestamp", "Shift", "Total Inspected", "OK", "NOT OK", "Efficiency"])

    # Updated read_csv to skip rows with issues
    df = pd.read_csv(LOG_CSV, header=None, names=["Timestamp", "ROI1_Status", "ROI1_Result", "ROI2_Status", "ROI2_Result"], on_bad_lines='skip')
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Define shift intervals
    shifts = {
        "Shift A": (dt.time(0, 0), dt.time(7, 0)),
        "Shift B": (dt.time(7, 0), dt.time(16, 0)),
        "Shift C": (dt.time(16, 0), dt.time(23, 59, 59)),
    }

    results = []
    
    for shift, (start, end) in shifts.items():
        # Filter data within shift times
        shift_data = df[(df["Timestamp"].dt.time >= start) & (df["Timestamp"].dt.time <= end)]
        
        total_inspected = len(shift_data)
        ok_count = shift_data.apply(lambda row: row.isin(["OK"]).sum(), axis=1).sum()
        not_ok_count = total_inspected * 2 - ok_count  # Assumes 2 ROIs per inspection
        efficiency = (ok_count / (total_inspected * 2) * 100) if total_inspected > 0 else 0
        
        results.append({
            "Shift": shift,
            "Total Inspected": total_inspected,
            "OK": ok_count,
            "NOT OK": not_ok_count,
            "Efficiency": f"{efficiency:.2f}%"
        })

    # Add overall data
    total_inspected = sum([r["Total Inspected"] for r in results])
    total_ok = sum([r["OK"] for r in results])
    total_not_ok = sum([r["NOT OK"] for r in results])
    overall_efficiency = (total_ok / (total_inspected * 2) * 100) if total_inspected > 0 else 0

    results.append({
        "Shift": "Overall",
        "Total Inspected": total_inspected,
        "OK": total_ok,
        "NOT OK": total_not_ok,
        "Efficiency": f"{overall_efficiency:.2f}%"
    })

    return pd.DataFrame(results)

def display_shift_table(parent):
    shift_data = get_shift_data()

    if shift_data.empty:
        print("No data to display in the table.")  # Debug print
        return  # Exit if there's no data

    # Create a treeview widget for displaying the table
    columns = ["Shift", "Total Inspected", "OK", "NOT OK", "Efficiency"]
    tree = ttk.Treeview(parent, columns=columns, show="headings")
    tree.pack(padx=10, pady=10, fill='both', expand=True)  # Ensure it fills the parent space

    # Set column headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=120)  # Adjust column width as needed

    # Insert data into the table
    for _, row in shift_data.iterrows():
        tree.insert("", "end", values=row.tolist())


class VisionInspectionUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Vision Inspection System")

        global roi_color
        roi_color = {f'ROI{i+1}': (255, 0, 0) for i in range(5)}  # Initialize colors for 5 ROIs

        self.video_fps = 5
        self.frame_delay = int(1000 / self.video_fps)  # Convert FPS to milliseconds delay

        self.calibration_label = None

        # Initialize essential attributes
        self.assembly_status = ""  # Initial status as an empty string
        self.status_color = (255, 255, 255)  # Default to white color

        # Track shift start and end
        self.current_shift_start = None
        self.current_shift_end = None

        # Enable full-screen mode
        self.master.attributes('-fullscreen', True)

        # Escape key to exit full-screen mode
        self.master.bind("<Escape>", self.exit_fullscreen)

        # Bind space key to trigger manual inspection
        self.master.bind("<a>", lambda event: self.manual_trigger())

        self.master.bind("<Control-t>", self.capture_reference_shortcut)
        # Add these bindings after other initializations
        self.master.bind("<Control-6>", lambda e: self.capture_multiple_roi_shortcut(["ROI1"]))
        self.master.bind("<Control-7>", lambda e: self.capture_multiple_roi_shortcut(["ROI2"]))
        self.master.bind("<Control-Alt-6>", lambda e: self.capture_multiple_not_ok_shortcut(["ROI1"]))
        self.master.bind("<Control-Alt-7>", lambda e: self.capture_multiple_not_ok_shortcut(["ROI2"]))

        # Initialize notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # self.setup_usb_button_listener(com_port="COM3")
        # Initialize PLC
        self.plc_conn = plc.IS_CONNECTED
        self.plc_type = plc.PLC_TYPE
        self.plc_config = plc.PLC_CONFIG[self.plc_type]
        self.plc = None
        self.reconnecting = False

        if plc.IS_CONNECTED:
            config = plc.PLC_CONFIG[self.plc_type]
            if self.plc_type == "Siemens":
                self.plc = plc.SiemensPLC(
                    config["ip"], config["rack"], config["slot"],
                    config["db_number"], config["communication_offset"],
                    config["heartbeat_offset"], config["status_offset"], config["keyword_offset"]
                )
            self.connect_plc()
        else:
            print("PLC Connection was not established")
            self.plc_conn = False

        # Set up the main dashboard before updating admin mode
        self.setup_dashboard()  # Ensures self.gauge_canvas is initialized

        #self.check_calibration_due(force_check=True)

        # Initialize gauge image placeholder
        create_efficiency_gauge(self.gauge_canvas, 0)  # Initialize with 0% efficiency

        # Add the Tuning tab and password-check behavior
        self.create_tuning_tab()
        self.setup_calibration_tab()

        # Bind event to check access when Tuning tab is selected
        self.notebook.bind("<<NotebookTabChanged>>", self.check_tuning_access)

        # Start the video feed only once
        self.start_camera_feed()

        # Start real-time updates for date, time, and shift resets
        self.update_date_time()
        self.schedule_shift_reset()

        # Now update the admin mode display after setting up the dashboard
        self.update_admin_mode_display()
        self.update_plc_status_display()
        self.master.focus_force()
        self.master.lift()
        self.master.update()
        self.start_heartbeat()

    def connect_plc(self):
        success = False
        while not success:
            try:
                success = self.plc.connect()
            except Exception as exc:
                err_msg = str(exc)
                if "Connection reset by peer" in err_msg or "ISO : An error occurred during send" in err_msg:
                    print(f"Connection reset during PLC connect attempt: {exc}")
                else:
                    print(f"PLC connect error: {exc}")
                success = False
                time.sleep(1)
        
        self.plc_conn = success
        self.update_plc_status_display()
        if success:
            #messagebox.showinfo("PLC Status", f"Connected to {self.plc_type} PLC")
            self.start_plc_listener()
        #else:
            #messagebox.showerror("PLC Status", f"Failed to connect to {self.plc_type} PLC")

    def disconnect_plc(self):
        if plc.IS_CONNECTED:
            if self.plc_conn:
                self.plc.disconnect()
                self.plc_conn = plc.IS_CONNECTED  # Fix: Set to False on disconnect
                self.update_plc_status_display()
                messagebox.showinfo("PLC Status", f"Disconnected from {self.plc_type} PLC")
            else:
                messagebox.showinfo("PLC Status", "PLC is already disconnected")
        else:
            messagebox.showerror("PLC Error", "PLC module is not available or not properly configured")

    def send_ok_status(self):
        if plc.IS_CONNECTED:
            if self.plc_conn:
                success = self.plc.send_status(1)
                if success:
                    messagebox.showinfo("PLC Status", "OK status sent to PLC")
                else:
                    self.plc_conn = False
                    self.update_plc_status_display()
                    if not self.reconnecting:
                        self.reconnecting = True
                        threading.Thread(target=self.reconnect_plc, daemon=True).start()
                    messagebox.showerror("PLC Error", "Failed to send OK status")
            else:
                messagebox.showerror("PLC Error", "PLC not connected. Please connect first.")
        else:
            messagebox.showerror("PLC Error", "PLC module is not available or not properly configured")

    def send_not_ok_status(self):
        if plc.IS_CONNECTED:
            if self.plc_conn:
                success = self.plc.send_status(2)
                if success:
                    messagebox.showinfo("PLC Status", "NOT OK status sent to PLC")
                else:
                    self.plc_conn = False
                    self.update_plc_status_display()
                    if not self.reconnecting:
                        self.reconnecting = True
                        threading.Thread(target=self.reconnect_plc, daemon=True).start()
                    messagebox.showerror("PLC Error", "Failed to send NOT OK status")
            else:
                messagebox.showerror("PLC Error", "PLC not connected. Please connect first.")
        else:
            messagebox.showerror("PLC Error", "PLC module is not available or not properly configured")
    def start_plc_listener(self):
        def listener():
            while True:  # Outer loop to handle reconnections
                if not plc.IS_CONNECTED or not self.plc_conn:
                    time.sleep(1)
                    continue
                try:
                    while plc.IS_CONNECTED and self.plc_conn:
                        keyword = self.plc.read_keyword()
                        if keyword == 'A':
                            print("Received keyword 'A' from PLC, triggering manual_trigger immediately")
                            # Schedule manual_trigger to run on the main thread ASAP
                            self.master.after(0, self.manual_trigger)
                        time.sleep(1)
                except Exception as exc:
                    print(f"PLC listener error: {exc}")
                    self.plc_conn = False
                    self.update_plc_status_display()
                    # Attempt reconnection
                    self.connect_plc()  # This will block until reconnected, then restart listening
        if plc.IS_CONNECTED:
            self.plc_listener_thread = threading.Thread(target=listener, daemon=True)
            self.plc_listener_thread.start()
        else:
            print("PLC module not available - listener not started")

    def save_triggered_images(self, frame, roi_statuses):
        triggered_base_folder = os.path.join(BASE_PATH, "Triggered_Images")
        os.makedirs(triggered_base_folder, exist_ok=True)
        cutoff_date = datetime.now() - pd.Timedelta(days=90)
        for roi_folder in os.listdir(triggered_base_folder):
            full_path = os.path.join(triggered_base_folder, roi_folder)
            if os.path.isdir(full_path):
                for image_file in os.listdir(full_path):
                    file_path = os.path.join(full_path, image_file)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_date:
                            os.remove(file_path)
                            print(f"Deleted old image: {file_path}")
                    except Exception as e:
                        print(f"Error checking file {file_path}: {e}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for roi, coords in roi_coordinates.items():
            status = roi_statuses.get(roi, "NOK")
            status_prefix = "OK" if status == "OK" else "NOK"
            roi_folder = os.path.join(triggered_base_folder, f"Triggered_{roi}")
            os.makedirs(roi_folder, exist_ok=True)
            cropped = frame[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
            if cropped.size == 0:
                print(f"Skipped saving empty cropped image for {roi}")
                continue
            filename = os.path.join(roi_folder, f"{status_prefix}_{timestamp}.jpg")
            cv2.imwrite(filename, cropped)

    def check_calibration_due(self, force_check=False):
        """Check if calibration is due and display a reminder if needed."""
        calibration_csv = 'calibration_dates.csv'  # File to store calibration date
        if not os.path.exists(calibration_csv):
            logging.info("Calibration CSV not found. Initializing with today's date.")
            self.initialize_calibration_date()

        # Load calibration date
        calibration_data = pd.read_csv(calibration_csv)
        last_calibration_date = datetime.strptime(calibration_data.iloc[0]['Last Calibration Date'], '%Y-%m-%d').date()
        next_calibration_date = last_calibration_date + pd.Timedelta(days=30)
        today = datetime.now().date()

        # Update calibration info on the dashboard
        if self.calibration_label:
            self.calibration_label.config(
                text=f"Last Calibration Date: {last_calibration_date}\nNext Calibration Date: {next_calibration_date}"
            )

        # Check if calibration is due Removed the popup reminder as per AL team request
        # if next_calibration_date <= today and (force_check or datetime.now().time().hour == 7):
        #     messagebox.showwarning(
        #         "Calibration Reminder",
        #         f"Calibration is due!\n\nLast Calibration Date: {last_calibration_date}\n"
        #         f"Next Calibration Date: {next_calibration_date}"
        #     )


    def update_calibration_date(self):
        """Update the last calibration date in the CSV."""
        calibration_csv = 'calibration_dates.csv'
        today = datetime.now().date()

        # Save the updated date back to the CSV
        calibration_data = pd.DataFrame({'Last Calibration Date': [today.strftime('%Y-%m-%d')]})
        calibration_data.to_csv(calibration_csv, index=False)

        # Recalculate the next calibration date
        next_calibration_date = today + pd.Timedelta(days=30)

        # Update the dashboard with new dates
        if self.calibration_label:
            self.calibration_label.config(
                text=f"Last Calibration Date: {today}\nNext Calibration Date: {next_calibration_date}"
            )

        messagebox.showinfo("Calibration Update", f"Calibration date updated to {today}.")



    def initialize_calibration_date(self):
        """Initialize the calibration date CSV with today's date."""
        calibration_csv = 'calibration_dates.csv'
        today = datetime.now().date()

        # Save today's date as the initial calibration date
        calibration_data = pd.DataFrame({'Last Calibration Date': [today.strftime('%Y-%m-%d')]})
        calibration_data.to_csv(calibration_csv, index=False)

        logging.info(f"Calibration date initialized to {today}.")

    def flash_roi_capture_success(self, roi, color=(0, 255, 0)):
        """Visual feedback for successful capture with customizable color"""
        original_color = roi_color[roi]
        roi_color[roi] = color
        self.master.after(500, lambda: self.reset_roi_color(roi, original_color))

    def reset_roi_color(self, roi, original_color):
        """Reset ROI color after flash"""
        roi_color[roi] = original_color

    def capture_not_ok_reference(self, roi, silent=False):
        """Capture a NOT_OK reference image for a specified ROI."""
        folder = os.path.join(ROI_FOLDERS.get(roi, f"{roi}_Images"), "NOT_OK")
        os.makedirs(folder, exist_ok=True)  # Ensure NOT_OK folder exists
        
        ret, frame = cap.read()
        if not ret:
            if not silent:
                messagebox.showerror("Error", "Unable to access video frame.")
            return

        coords = roi_coordinates[roi]
        cropped = frame[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        
        if cropped.size == 0:
            if not silent:
                messagebox.showerror("Error", f"No valid region to save for {roi}.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{folder}/{timestamp}_NOT_OK.jpg"
        cv2.imwrite(filename, cropped)

    def capture_multiple_roi_shortcut(self, rois, event=None):
        """Shortcut for capturing OK reference images for multiple ROIs silently."""
        for roi in rois:
            print("Available ROIs:", list(roi_coordinates.keys()))
            # print(roi)
            # print(roi_coordinates)
            if roi in roi_coordinates:
                self.capture_reference_image(roi, silent=True)
                self.flash_roi_capture_success(roi)
                print(f"OK reference image captured for {roi}")


    def capture_multiple_not_ok_shortcut(self, rois, event=None):
        print("Available ROIs:", list(roi_coordinates.keys()))
        """Shortcut for capturing NOT_OK reference images for multiple ROIs silently."""
        for roi in rois:
            if roi in roi_coordinates:
                self.capture_not_ok_reference(roi, silent=True)
                self.flash_roi_capture_success(roi, color=(255, 0, 0))  # Red flash for NOT_OK
                print(f"NOT_OK reference image captured for {roi}")

    def capture_reference_shortcut(self, rois, event=None):
        """Shortcut for capturing reference images for multiple ROIs silently."""
        for roi in rois:
            if roi in roi_coordinates:
                self.capture_reference_image(roi, silent=True)
                self.flash_roi_capture_success(roi)
                print(f"Reference image captured for {roi}")



    def update_date_time(self):
        """Update the date and time in real-time."""
        current_time = datetime.now()
        self.date_time_label.config(text=f"Date & Time: {current_time:%Y-%m-%d %H:%M:%S}")

        # Check calibration due at 7:00 AM
        if current_time.time().hour == 7 and current_time.time().minute == 0:
            #self.check_calibration_due()
            pass

        self.master.after(1000, self.update_date_time)  # Update every second


    def schedule_shift_reset(self):
        """Schedule the reset of counters and efficiency at the end of each shift."""
        now = datetime.now()
        shifts = [
            {"start": dt.time(7, 30), "end": dt.time(15, 30)},  # Shift A
            {"start": dt.time(15, 30), "end": dt.time(23, 59)},  # Shift B
            {"start": dt.time(0, 0), "end": dt.time(7, 30)}  # Shift C
        ]

        for shift in shifts:
            start_time = datetime.combine(now.date(), shift["start"])
            end_time = datetime.combine(now.date(), shift["end"])
            if shift["start"] > shift["end"]:
                # Handle overnight shift (e.g., 12:00 AM to 7:30 AM)
                end_time += timedelta(days=1)
            if start_time <= now < end_time:
                self.current_shift_start = start_time
                self.current_shift_end = end_time

        if self.current_shift_end:
            seconds_to_reset = (self.current_shift_end - now).total_seconds()
            self.master.after(int(seconds_to_reset * 1000), self.reset_shift_counters)

    def reset_shift_counters(self):
        """Reset counters at the end of each shift and display summary."""
        global total_inspected, first_pass_right, trigger_count, ok_count

        # Calculate and log shift efficiency
        efficiency = (ok_count / trigger_count * 100) if trigger_count > 0 else 0
        shift_summary = (
            f"Shift Ended\n\n"
            f"Total Inspected: {total_inspected}\n"
            f"First Pass Right: {first_pass_right}\n"
            f"Efficiency: {efficiency:.2f}%"
        )
        logging.info(shift_summary)  # Log the summary instead of showing a pop-up

        # Optionally display the summary in the dashboard (e.g., a label or text widget)
        self.shift_summary_label.config(text=shift_summary)

        # Reset counters
        total_inspected = 0
        first_pass_right = 0
        trigger_count = 0
        ok_count = 0

        # Reset labels
        self.total_inspected_label.config(text="Total Inspected:\n\n0")
        self.first_pass_label.config(text="First Pass Right:\n\n0")
        self.efficiency_label.config(text="Efficiency: 0%")
        create_efficiency_gauge(self.gauge_canvas, 0)

        # Schedule the next reset
        self.schedule_shift_reset()

    # def listen_to_usb_button(self):
    #     """Continuously listen for USB button signals."""
    #     print("Starting USB button listener...")
    #     try:
    #         while True:
    #             serial_conn = self.serial_conn  # Use a local reference
    #             if not serial_conn or not serial_conn.is_open:
    #                 print("Serial connection is not available or closed.")
    #                 break

    #             try:
    #                 data = serial_conn.readline().decode('utf-8', errors='ignore').strip()
    #                 if data:
    #                     print(f"Raw Data Received: {data}")
    #                     if data == "A":  # Match the button's signal
    #                         print("Button pressed!")
    #                         self.manual_trigger()  # Call the manual trigger function
    #                     else:
    #                         print(f"Unexpected data: {data}")
    #             except serial.SerialException as e:
    #                 print(f"SerialException occurred: {e}")
    #                 break  # Exit the loop on serial exception
    #             except AttributeError as e:
    #                 print(f"AttributeError: {e} (self.serial_conn might be None)")
    #                 break  # Exit the loop on attribute error
    #     except Exception as e:
    #         print(f"Unhandled error in USB button listener: {e}")


    # def setup_usb_button_listener(self, com_port="COM3", baudrate=9600):
    #     """Initialize and listen to USB button presses."""
    #     try:
    #         self.serial_conn = serial.Serial(com_port, baudrate, timeout=1)
    #         if self.serial_conn.is_open:
    #             print(f"Connected to {com_port} successfully.")
    #             self.usb_button_thread = threading.Thread(target=self.listen_to_usb_button, daemon=True)
    #             self.usb_button_thread.start()
    #             print(f"Listening for USB button on {com_port}...")
    #         else:
    #             print(f"Failed to open serial connection on {com_port}.")
    #             self.serial_conn = None
    #     except Exception as e:
    #         print(f"Error initializing USB button listener: {e}")
    #         self.serial_conn = None


    def prompt_select_roi(self):
        """Prompt the user to select an ROI."""
        rois = list(roi_coordinates.keys())
        selected_roi = simpledialog.askstring("Select ROI", f"Enter ROI name ({', '.join(rois)}) for optimization:")
        if selected_roi in rois:
            return selected_roi
        else:
            messagebox.showerror("Error", "Invalid ROI name.")
            return None

    def setup_dashboard(self):
        """Sets up the Dashboard tab with an image display, 4 info boxes, and a gauge."""
        global cap

        # Release the camera if it was initialized
        if cap is not None and cap.isOpened():
            cap.release()
            cap = None  # Ensure the camera resource is released

        self.dashboard_tab = tk.Frame(self.notebook, bg="#f0f0f0")  # Light background for app-like feel
        self.notebook.add(self.dashboard_tab, text="Dashboard")

        # Header Area
        header_frame = tk.Frame(self.dashboard_tab, bg="#4169E1", height=80)  # Royal Blue background
        header_frame.pack(side="top", fill="x")
        header_frame.pack_propagate(False)  # Prevent the frame from resizing based on its content

        # Left Logo
        try:
            left_logo_path = "TRUCK_DIGITAL.png"  # Replace with the correct path
            left_logo = Image.open(left_logo_path)
            left_logo = left_logo.resize((229, 80), Image.Resampling.LANCZOS)
            self.left_logo_tk = ImageTk.PhotoImage(left_logo)
        except Exception as e:
            print(f"Error loading left logo: {e}")
            self.left_logo_tk = None

        if self.left_logo_tk:
            left_logo_label = tk.Label(header_frame, image=self.left_logo_tk, bg="#4169E1")
            left_logo_label.pack(side="left", anchor="w", padx=0)

        # Center Title
        title_label = tk.Label(
            header_frame,
            text="Smart Digital Solutions - Oil Cooler Gasket Detection",
            font=("Arial", 24, "bold"),
            bg="#4169E1",
            fg="white"
        )
        title_label.pack(side="left", expand=True, padx=10)

        # Right Logo
        try:
            right_logo_path = "logo.png"  # Replace with the correct path
            right_logo = Image.open(right_logo_path)
            right_logo = right_logo.resize((234, 68), Image.Resampling.LANCZOS)
            self.right_logo_tk = ImageTk.PhotoImage(right_logo)
        except Exception as e:
            print(f"Error loading right logo: {e}")
            self.right_logo_tk = None

        if self.right_logo_tk:
            right_logo_label = tk.Label(header_frame, image=self.right_logo_tk, bg="#4169E1")
            right_logo_label.pack(side="right", anchor="e", padx=0)

        # Main Content Area
        content_frame = tk.Frame(self.dashboard_tab, bg="#f0f0f0")
        content_frame.pack(fill="both", expand=True, pady=10)

        # Video Feed and Right Image Section
        video_photo_frame = tk.Frame(content_frame, bg="#f0f0f0")
        video_photo_frame.pack(side="top", pady=5)

        # Video Feed Frame
        video_frame = tk.Frame(video_photo_frame, bg="#f0f0f0", width=640, height=420)
        video_frame.grid(row=0, column=0, padx=5)

        self.camera_label = tk.Label(video_frame, text="Camera Feed Not Available", font=("Arial", 16), bg="#f0f0f0", fg="red", relief="solid", bd=1)
        self.camera_label.pack(fill="both", expand=True)

        # Right Image Frame
        photo_frame = tk.Frame(video_photo_frame, bg="#f0f0f0", width=640, height=420)
        photo_frame.grid(row=0, column=1, padx=5)

        try:
            photo_path = "OK.png"  # Replace with your photo file path
            right_photo = Image.open(photo_path)
            right_photo = right_photo.resize((640, 420), Image.Resampling.LANCZOS)
            self.right_photo_tk = ImageTk.PhotoImage(right_photo)
        except Exception as e:
            print(f"Error loading right photo: {e}")
            self.right_photo_tk = None

        if self.right_photo_tk:
            right_photo_label = tk.Label(photo_frame, image=self.right_photo_tk, relief="solid", bd=1)
            right_photo_label.pack(fill="both", expand=True)
        else:
            right_photo_label = tk.Label(photo_frame, text="Right Photo Not Found", font=("Arial", 16), bg="#f0f0f0", fg="red", relief="solid", bd=1)
            right_photo_label.pack()

        # Info Boxes and Gauge Section
        info_gauge_frame = tk.Frame(content_frame, bg="#f0f0f0")
        info_gauge_frame.pack(side="top", pady=10, fill="x")

        # Info Boxes Frame
        info_boxes_frame = tk.Frame(info_gauge_frame, bg="#f0f0f0")
        info_boxes_frame.pack(side="left", padx=10)

      # Create Info Boxes
        self.date_time_label = tk.Label(
            info_boxes_frame,
            text=f"Date & Time: {datetime.now():%Y-%m-%d %H:%M:%S}",
            font=("Arial", 12),
            bg="#f0f0f0",
            relief="ridge",
            bd=2,
            width=48
        )
        self.date_time_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Add to setup_dashboard method
        self.calibration_label = tk.Label(
            info_boxes_frame,
            text="Last Calibration Date: \nNext Calibration Date: ",
            font=("Arial", 12),
            bg="#f0f0f0",
            relief="ridge",
            bd=2,
            width=48
        )
        self.calibration_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        plc_status_text = f"PLC: {self.plc_type}\nStatus: {'Connected' if self.plc_conn else 'Disconnected'}"
        self.plc_status_label = tk.Label(
            info_boxes_frame,
            text=plc_status_text,
            font=("Arial", 12),
            bg="#ccffcc" if self.plc_conn else "#ffcccc",
            relief="ridge",
            bd=2,
            width=48
        )
        self.plc_status_label.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.total_inspected_label = tk.Label(
            info_boxes_frame,
            text="Total Inspected:\n\n0",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            relief="ridge",
            bd=2,
            width=48
        )
        self.total_inspected_label.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.first_pass_label = tk.Label(
            info_boxes_frame,
            text="First Pass Right:\n\n0",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            relief="ridge",
            bd=2,
            width=48
        )
        self.first_pass_label.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.shift_summary_label = tk.Label(
            info_boxes_frame,
            text="Shift Summary:\n\n",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            relief="ridge",
            bd=2,
            width=48
        )
        self.shift_summary_label.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        # Gauge Frame
        gauge_frame = tk.Frame(info_gauge_frame, bg="#f0f0f0")
        gauge_frame.pack(side="right", padx=80, pady=20)

        # Gauge Display
        self.gauge_canvas = tk.Canvas(gauge_frame, width=200, height=100, bg="#f0f0f0", highlightthickness=0)
        self.gauge_canvas.pack()

        # Initialize gauge with 0% efficiency
        create_efficiency_gauge(self.gauge_canvas, 0)

        # Efficiency Label
        self.efficiency_label = tk.Label(
            gauge_frame,
            text="Efficiency: 0%",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="black"
        )
        self.efficiency_label.pack(pady=10)

        # Trigger Count Label
        self.trigger_count_label = tk.Label(
            gauge_frame,
            text=" ",
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="black"
        )
        self.trigger_count_label.pack(pady=5)

                
    def create_tuning_tab(self):
        """Create a Tuning tab that prompts for a password to access admin mode."""
        self.tuning_tab = tk.Frame(self.notebook)
        self.notebook.add(self.tuning_tab, text="Tuning")

        # Add a label and button to trigger admin mode password prompt
        tk.Label(self.tuning_tab, text="Admin Access Required", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.tuning_tab, text="Enter Password for Admin Access", command=self.enable_admin_mode).pack(pady=10)

    def check_tuning_access(self, event=None):
        """Check if Tuning tab was selected and prompt for password if not in admin mode."""
        # Identify the selected tab
        selected_tab = self.notebook.select()
        selected_tab_text = self.notebook.tab(selected_tab, "text")
        
        # If "Tuning" is selected and not in admin mode, ask for password
        if selected_tab_text == "Tuning" and not admin_mode:
            self.enable_admin_mode()

            # If admin mode is not enabled (wrong password), switch back to the Dashboard tab
            if not admin_mode:
                self.notebook.select(self.dashboard_tab)

    def setup_threshold_settings(self):
        """Setup the threshold settings tab with sliders for zoom and focus applied uniformly across all ROIs."""
        self.threshold_tab = tk.Frame(self.notebook)
        self.notebook.add(self.threshold_tab, text="Threshold Settings")

        # Configure layout
        self.threshold_tab.grid_columnconfigure(0, weight=1)

        content_frame = tk.Frame(self.threshold_tab)
        content_frame.place(relx=0.5, rely=0.5, anchor="center")

        # ROI count input
        tk.Label(content_frame, text="Set Number of ROIs (1-7):", font=("Arial", 16)).grid(row=0, column=0, pady=20)
        self.num_rois_entry = tk.Entry(content_frame, font=("Arial", 16), width=10)
        self.num_rois_entry.insert(0, str(len(roi_coordinates)))
        self.num_rois_entry.grid(row=0, column=1, pady=20)

        # Save ROI Count button
        self.save_rois_button = tk.Button(
            content_frame,
            text="Save ROI Count",
            command=self.save_roi_count,
            font=("Arial", 16, "bold"),
            bg="#4CAF50",
            fg="black",
            activebackground="#45a049",
            relief="raised",
            bd=2,
            padx=10,
            pady=8,
            width=20,
        )
        self.save_rois_button.grid(row=0, column=2, padx=30, pady=20)

        # Threshold entries for individual ROIs
        self.threshold_entries = {}
        row_index = 1
        for roi, data in thresholds.items():
            tk.Label(content_frame, text=f"{data['Display Name']} Threshold:", font=("Arial", 16)).grid(row=row_index, column=0, pady=10)
            entry = tk.Entry(content_frame, font=("Arial", 16), width=10)
            entry.insert(0, data['Threshold'])
            entry.grid(row=row_index, column=1, pady=10)
            self.threshold_entries[roi] = entry
            row_index += 1

        # Global Zoom and Focus Sliders (applied to all ROIs)
        # Load initial values from the first ROI in the dictionary
        first_roi = next(iter(thresholds.values()), {})
        initial_zoom = first_roi.get('Zoom', 1.0)
        initial_focus = first_roi.get('Focus', 0)

        tk.Label(content_frame, text="Zoom (applies to all ROIs):", font=("Arial", 16)).grid(row=row_index, column=0, pady=10)
        self.zoom_slider = tk.Scale(content_frame, from_=1, to=5, resolution=0.1, orient="horizontal", length=200)
        self.zoom_slider.set(initial_zoom)  # Initialize with value from CSV
        self.zoom_slider.grid(row=row_index, column=1, pady=10)

        tk.Label(content_frame, text="Focus (applies to all ROIs):", font=("Arial", 16)).grid(row=row_index + 1, column=0, pady=10)
        self.focus_slider = tk.Scale(content_frame, from_=0, to=1000, orient="horizontal", length=200)
        self.focus_slider.set(initial_focus)  # Initialize with value from CSV
        self.focus_slider.grid(row=row_index + 1, column=1, pady=10)

        # Save Button
        tk.Button(content_frame, text="Save Thresholds", command=self.save_thresholds_ui, font=("Arial", 16)).grid(
            row=row_index + 2, column=0, columnspan=2, pady=20
        )




    def setup_roi_management(self):
        """Setup the ROI management tab with organized button grouping and larger buttons and boxes."""
        self.roi_management_tab = tk.Frame(self.notebook)
        self.notebook.add(self.roi_management_tab, text="PLC & ROI Management")

        # Create a canvas and scrollbar
        canvas = tk.Canvas(self.roi_management_tab, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.roi_management_tab, orient="vertical", command=canvas.yview)
        # scrollable_frame = tk.Frame(canvas)

        # # Configure the canvas
        # scrollable_frame.bind(
        #     "<Configure>",
        #     lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        # )
        # canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        # canvas.pack(side="left", fill="both", expand=True)

        # # Add mousewheel scrolling
        # def _on_mousewheel(event):
        #     canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        # canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Center the content vertically and horizontally
        # content_frame = tk.Frame(scrollable_frame)
        # content_frame.pack(expand=True, padx=20, pady=20)  # Center-align content
        scrollable_container = tk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_container, anchor="n", width=self.roi_management_tab.winfo_screenwidth())
        scrollable_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        def on_mouse_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        tk.Label(scrollable_container, text="PLC Control", font=("Arial", 24, "bold"), pady=20).pack()
        plc_frame = tk.LabelFrame(scrollable_container, text="PLC Actions", font=("Arial", 18, "bold"), padx=20, pady=15, relief="solid")
        plc_frame.pack(pady=15, padx=50)
        tk.Button(
            plc_frame,
            text="Connect PLC",
            command=self.connect_plc,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="black",
            activebackground="#45a049",
            relief="raised",
            bd=3,
            padx=15,
            pady=8
        ).pack(side="left", padx=10)
        tk.Button(
            plc_frame,
            text="Disconnect PLC",
            command=self.disconnect_plc,
            font=("Arial", 14, "bold"),
            bg="#FF5722",
            fg="black",
            activebackground="#E64A19",
            relief="raised",
            bd=3,
            padx=15,
            pady=8
        ).pack(side="left", padx=10)
        tk.Button(
            plc_frame,
            text="Send OK",
            command=self.send_ok_status,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="black",
            activebackground="#45a049",
            relief="raised",
            bd=3,
            padx=15,
            pady=8
        ).pack(side="left", padx=10)
        tk.Button(
            plc_frame,
            text="Send NOT OK",
            command=self.send_not_ok_status,
            font=("Arial", 14, "bold"),
            bg="#FF5722",
            fg="black",
            activebackground="#E64A19",
            relief="raised",
            bd=3,
            padx=15,
            pady=8
        ).pack(side="left", padx=10)
        tk.Label(scrollable_container, text="ROI Management", font=("Arial", 24, "bold"), pady=20).pack()

        # # Section Title
        # tk.Label(
        #     content_frame,
        #     text="ROI Management",
        #     font=("Arial", 24, "bold"),
        #     pady=20
        # ).grid(row=0, column=0, columnspan=2, pady=30)  # Centered title with more padding

        # Dictionaries to store buttons for each ROI
        self.rename_buttons = {}
        self.mark_buttons = {}
        self.capture_buttons = {}
        self.reset_buttons = {}

        # Add ROI buttons in two columns
        column_index = 0
        row_index = 1
        self.set_ref_buttons = {}
        roi_list = list(roi_coordinates.keys())
        for i in range(0, len(roi_list), 2):
            row_wrapper = tk.Frame(scrollable_container)
            row_wrapper.pack(pady=15, anchor="center")
            for j in range(2):
                if i + j < len(roi_list):
                    roi = roi_list[i + j]
                    roi_frame = tk.LabelFrame(
                        row_wrapper,
                        text=f"{roi} Actions",
                        font=("Arial", 18, "bold"),
                        padx=20,
                        pady=20,
                        relief="solid",
                        width=345,
                        height=430
                    )
                    roi_frame.pack(side="left", padx=30)
                    roi_frame.grid_propagate(False)
                    self.rename_buttons[roi] = tk.Button(
                        roi_frame, text="Rename ROI", command=lambda r=roi: self.rename_roi(r),
                        font=("Arial", 16, "bold"), bg="#4CAF50", fg="black",
                        activebackground="#45a049", relief="raised", bd=3,
                        padx=15, pady=12, width=18
                    )
                    self.rename_buttons[roi].pack(pady=10)
                    self.mark_buttons[roi] = tk.Button(
                        roi_frame, text="Set ROI", command=lambda r=roi: self.set_roi(r),
                        font=("Arial", 16, "bold"), bg="#2196F3", fg="black",
                        activebackground="#1976D2", relief="raised", bd=3,
                        padx=15, pady=12, width=18
                    )
                    self.mark_buttons[roi].pack(pady=10)
                    self.capture_buttons[roi] = tk.Button(
                        roi_frame, text="Capture Reference", command=lambda r=roi: self.capture_reference_image(r),
                        font=("Arial", 16, "bold"), bg="#FFC107", fg="black",
                        activebackground="#FFA000", relief="raised", bd=3,
                        padx=15, pady=12, width=18
                    )
                    self.capture_buttons[roi].pack(pady=10)
                    self.set_ref_buttons[roi] = tk.Button(
                        roi_frame, text="Set Ref. ROI", command=lambda r=roi: [self.capture_snapshot(), self.set_reference_roi(r)],
                        font=("Arial", 16, "bold"), bg="#00BCD4", fg="black",
                        activebackground="#0097A7", relief="raised", bd=3,
                        padx=15, pady=12, width=18
                    )
                    self.set_ref_buttons[roi].pack(pady=10)
                    self.reset_buttons[roi] = tk.Button(
                        roi_frame, text="Reset References", command=lambda r=roi: self.reset_roi_references(r),
                        font=("Arial", 16, "bold"), bg="#F44336", fg="black",
                        activebackground="#D32F2F", relief="raised", bd=3,
                        padx=15, pady=12, width=18
                    )
                    self.reset_buttons[roi].pack(pady=10)

    def save_thresholds_ui(self):
        """Save threshold, zoom, and focus values to the thresholds dictionary and CSV."""
        for roi, entry in self.threshold_entries.items():
            thresholds[roi]['Threshold'] = float(entry.get())

        # Apply global zoom and focus to all ROIs
        zoom_value = self.zoom_slider.get()
        focus_value = self.focus_slider.get()
        for roi in thresholds:
            thresholds[roi]['Zoom'] = zoom_value
            thresholds[roi]['Focus'] = focus_value

        save_thresholds()
        messagebox.showinfo("Success", "Thresholds, Zoom, and Focus saved successfully.")


    def save_roi_count(self):
        """Save the number of ROIs and update the coordinates dynamically."""
        try:
            # Fetch the number of ROIs from the input
            num_rois = int(self.num_rois_entry.get())
            if 1 <= num_rois <= 7:
                # Update global variables related to ROIs
                global roi_coordinates, ROI_FOLDERS, thresholds

                # Update ROI coordinates
                roi_coordinates = {
                    f'ROI{i + 1}': {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'Display Name': f'ROI{i + 1}'}
                    for i in range(num_rois)
                }

                # Update ROI folders
                ROI_FOLDERS = {roi: os.path.join(BASE_PATH, f"{roi}_Images") for roi in roi_coordinates.keys()}
                
                # Update thresholds for new ROIs
                thresholds = {
                    roi: {'Threshold': 0.7, 'Display Name': roi, 'Zoom': 1.0, 'Focus': 0} for roi in roi_coordinates.keys()
                }

                # Ensure directories for ROI folders exist
                for folder in ROI_FOLDERS.values():
                    os.makedirs(folder, exist_ok=True)

                # Save updated ROI data and thresholds
                save_roi_coordinates()
                save_thresholds()

                # Refresh the UI to reflect the updated ROIs
                self.setup_threshold_settings()
                self.setup_roi_management()
                messagebox.showinfo("Success", f"Number of ROIs set to {num_rois}.")
            else:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number between 1 and 7.")


    def update_roi_coordinates(self, num_rois):
        global roi_coordinates, ROI_FOLDERS, roi_color, roi_status
        roi_coordinates = {f'ROI{i + 1}': {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'Display Name': f'ROI{i + 1}'} for i in range(num_rois)}
        
        # Use absolute paths for ROI_FOLDERS
        ROI_FOLDERS = {roi: os.path.join(BASE_PATH, f"{roi}_Images") for roi in roi_coordinates.keys()}
        for folder in ROI_FOLDERS.values():
            os.makedirs(folder, exist_ok=True)

        roi_color = {roi: (255, 0, 0) for roi in roi_coordinates.keys()}
        roi_status = {roi: "Waiting" for roi in roi_coordinates.keys()}

        save_roi_coordinates()
        self.setup_roi_management()

    def rename_roi(self, roi):
        new_display_name = tk.simpledialog.askstring("Rename Display Name", f"Enter display name for {roi}:")
        if new_display_name:
            roi_coordinates[roi]['Display Name'] = new_display_name
            save_roi_coordinates()
            self.setup_roi_management()
            messagebox.showinfo("Success", f"{roi} display name updated to {new_display_name}.")

    def set_roi(self, roi):
        """Set ROI with a confirmation popup before modifying existing reference images."""
        global cap

        # Check if existing references exist for this ROI
        folder = ROI_FOLDERS.get(roi, f"{roi}_Images")
        if os.path.exists(folder) and os.listdir(folder):
            confirm = messagebox.askyesno(
                "Warning", 
                f"Changing the ROI will make existing reference images unusable!\n\n"
                f"Are you sure you want to proceed?"
            )
            if not confirm:
                return  # Exit if user cancels

        # Apply the current zoom and focus settings
        roi_zoom = thresholds[roi].get('Zoom', 1.0)
        roi_focus = thresholds[roi].get('Focus', 1.0)
        cap.set(cv2.CAP_PROP_FOCUS, roi_focus)

        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Unable to access camera.")
            return

        # 🔄 Rotate the frame
        #frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Apply zoom for setting ROI
        height, width = frame.shape[:2]
        new_width, new_height = int(width / roi_zoom), int(height / roi_zoom)
        x_start = (width - new_width) // 2
        y_start = (height - new_height) // 2
        cropped_frame = frame[y_start:y_start + new_height, x_start:x_start + new_width]
        cropped_frame_resized = cv2.resize(cropped_frame, (640, 420))

        # Allow the user to select ROI on the zoomed frame
        roi_coords = cv2.selectROI("Set ROI", cropped_frame_resized, showCrosshair=True)
        cv2.destroyAllWindows()

        # Save ROI coordinates relative to the original frame
        scale_x = new_width / 640
        scale_y = new_height / 420
        x1 = int(roi_coords[0] * scale_x + x_start)
        y1 = int(roi_coords[1] * scale_y + y_start)
        x2 = int((roi_coords[0] + roi_coords[2]) * scale_x + x_start)
        y2 = int((roi_coords[1] + roi_coords[3]) * scale_y + y_start)
        roi_coordinates[roi] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'Display Name': roi}

        # # Save direct coordinates without any scaling
        # x1 = int(roi_coords[0])
        # y1 = int(roi_coords[1])
        # x2 = int(roi_coords[0] + roi_coords[2])
        # y2 = int(roi_coords[1] + roi_coords[3])
        
        # roi_coordinates[roi] = {
        #     'x1': x1, 
        #     'y1': y1, 
        #     'x2': x2, 
        #     'y2': y2, 
        #     'Display Name': roi
        # }

        save_roi_coordinates()





    def set_reference_roi(self, roi):
        frame = getattr(self, 'last_snapshot', None)
        if frame is None:
            messagebox.showerror("Error", "No snapshot available for setting reference ROI.")
            return
        zoom = thresholds[roi].get('Zoom', 1.0)
        height, width = frame.shape[:2]
        new_width, new_height = int(width / zoom), int(height / zoom)
        x_start = (width - new_width) // 2
        y_start = (height - new_height) // 2
        cropped = frame[y_start:y_start + new_height, x_start:x_start + new_width]
        resized = cv2.resize(cropped, (640, 420))
        roi_box = cv2.selectROI("Set Ref. ROI", resized, showCrosshair=True)
        cv2.destroyAllWindows()
        scale_x = new_width / 640
        scale_y = new_height / 420
        x1 = int(roi_box[0] * scale_x + x_start)
        y1 = int(roi_box[1] * scale_y + y_start)
        x2 = int((roi_box[0] + roi_box[2]) * scale_x + x_start)
        y2 = int((roi_box[1] + roi_box[3]) * scale_y + y_start)
        reference_roi_coordinates[roi] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        messagebox.showinfo("Success", f"Reference ROI for {roi} saved.")

    def capture_reference_image(self, roi, silent=False):
        """Capture a reference image for a specified ROI."""
        folder = ROI_FOLDERS.get(roi, f"{roi}_Images")
        os.makedirs(folder, exist_ok=True)  # Ensure folder exists
        ret, frame = cap.read()
        if not ret:
            if not silent:
                messagebox.showerror("Error", "Unable to access camera.")
            return

        coords = roi_coordinates[roi]
        cropped = frame[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        
        # Check if cropped is empty before saving
        if cropped.size == 0:
            if not silent:
                messagebox.showerror("Error", f"No valid region to save for {roi}.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{folder}/{timestamp}.jpg"
        cv2.imwrite(filename, cropped)

        if not silent:
            messagebox.showinfo("Success", f"Captured and saved to {filename}.")

    def update_plc_status_display(self):
        if hasattr(self, 'plc_status_label'):
            status_text = f"PLC: {self.plc_type}\nStatus: {'Connected' if self.plc_conn else 'Disconnected'}"
            bg_color = "#ccffcc" if self.plc_conn else "#ffcccc"
            self.plc_status_label.config(text=status_text, bg=bg_color)

    def manual_trigger(self):
        global trigger_count, ok_count, total_inspected, first_pass_right

        ret, frame = cap.read()
        if not ret:
            print("Debug: Failed to read frame")
            return

        trigger_count += 1
        total_inspected += 1
        self.total_inspected_label.config(text=f"Total Inspected:\n\n{total_inspected}")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nDebug: Starting inspection #{total_inspected}")
        self.total_inspected_label.config(text=f"Total Inspected:\n\n{total_inspected}")
        roi_statuses = {}
        roi_accuracies = {}

        # Group ROIs
        h6_rois = ['ROI1']
        h4_rois = ['ROI2']
        
        # Get accuracies for each ROI
        for roi in roi_coordinates.keys():
            ok_folder = ROI_FOLDERS.get(roi, "")
            not_ok_folder = os.path.join(ok_folder, "NOT_OK")
            
            ok_accuracy = self.get_max_accuracy(frame, roi, ok_folder)
            not_ok_accuracy = self.get_max_accuracy(frame, roi, not_ok_folder)
            
            print(f"\nDebug: {roi} Accuracies:")
            print(f"OK accuracy: {ok_accuracy:.3f}")
            print(f"NOT_OK accuracy: {not_ok_accuracy:.3f}")

            if ok_accuracy > not_ok_accuracy:
                roi_statuses[roi] = "OK"
                roi_color[roi] = (0, 255, 0)
                print(f"{roi} Status: OK (OK > NOT_OK)")
            else:
                roi_statuses[roi] = "NOT OK"
                roi_color[roi] = (0, 0, 255)
                print(f"{roi} Status: NOT OK (OK <= NOT_OK)")

        # Check H6 status (ROI1, ROI2, ROI3)
        h6_all_ok = all(roi_statuses.get(roi) == "OK" for roi in h6_rois)
        h6_any_ok = any(roi_statuses.get(roi) == "OK" for roi in h6_rois)

        # Check H4 status (ROI4, ROI5)
        h4_all_ok = all(roi_statuses.get(roi) == "OK" for roi in h4_rois)
        h4_any_ok = any(roi_statuses.get(roi) == "OK" for roi in h4_rois)

        # Set assembly status based on conditions
        if h6_all_ok and h4_all_ok:
            self.assembly_status = "Assembly - OK"
            self.status_color = (0, 255, 0)
            ok_count += 1
            first_pass_right += 1
            if plc.IS_CONNECTED and self.plc_conn:
                success = self.plc.send_status(1) # Send OK status
                if not success:
                    self.plc_conn = False
                    self.update_plc_status_display()
                    if not self.reconnecting:
                        self.reconnecting = True
                        threading.Thread(target=self.reconnect_plc, daemon=True).start()

        elif h4_all_ok or h6_all_ok:
            self.assembly_status = "Assembly - NOT OK"
            self.status_color = (0, 0, 255)
            if plc.IS_CONNECTED and self.plc_conn:
                success = self.plc.send_status(2) # Send NOT OK status
                if not success:
                    self.plc_conn = False
                    self.update_plc_status_display()
                    if not self.reconnecting:
                        self.reconnecting = True
                        threading.Thread(target=self.reconnect_plc, daemon=True).start()  
        elif h6_any_ok and h4_any_ok:
            self.assembly_status = "Assembly - OK"
            self.status_color = (0, 255, 0)
            ok_count += 1
            first_pass_right += 1
            if plc.IS_CONNECTED and self.plc_conn:
                success = self.plc.send_status(1) # Send OK status
                if not success:
                    self.plc_conn = False
                    self.update_plc_status_display()
                    if not self.reconnecting:
                        self.reconnecting = True
                        threading.Thread(target=self.reconnect_plc, daemon=True).start()

        elif h4_any_ok or h6_any_ok:
            self.assembly_status = "Assembly - NOT OK"
            self.status_color = (0, 0, 255)
            if plc.IS_CONNECTED and self.plc_conn:
                success = self.plc.send_status(2) # Send NOT OK status
                if not success:
                    self.plc_conn = False
                    self.update_plc_status_display()
                    if not self.reconnecting:
                        self.reconnecting = True
                        threading.Thread(target=self.reconnect_plc, daemon=True).start()  

        else:
            self.assembly_status = "Assembly - NOT OK"
            self.status_color = (0, 0, 255)
            if plc.IS_CONNECTED and self.plc_conn:
                success = self.plc.send_status(2) # Send NOT OK status
                if not success:
                    self.plc_conn = False
                    self.update_plc_status_display()
                    if not self.reconnecting:
                        self.reconnecting = True
                        threading.Thread(target=self.reconnect_plc, daemon=True).start()  


        # Update UI components
        self.first_pass_label.config(text=f"First Pass Right:\n\n{first_pass_right}")
        update_log(timestamp, roi_statuses)
        efficiency = (ok_count / trigger_count) * 100 if trigger_count else 0
        self.efficiency_label.config(text=f"Efficiency: {efficiency:.2f}%")
        create_efficiency_gauge(self.gauge_canvas, efficiency)
        self.last_snapshot = frame
        self.master.after(2000, self.reset_rois)



    def get_max_accuracy(self, frame, roi, folder):
        """Helper function to get maximum accuracy for a given folder"""
        max_accuracy = 0
        templates = [cv2.imread(f) for f in glob.glob(f"{folder}/*.jpg") if os.path.isfile(f)]
        coords = roi_coordinates[roi]
        cropped = frame[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        
        for template in templates:
            if template is not None and cropped.size > 0:
                if cropped.shape[0] < template.shape[0] or cropped.shape[1] < template.shape[1]:
                    template = cv2.resize(template, (cropped.shape[1], cropped.shape[0]))
                res = cv2.matchTemplate(cropped, template, cv2.TM_CCOEFF_NORMED)
                _, accuracy, _, _ = cv2.minMaxLoc(res)
                max_accuracy = max(max_accuracy, accuracy)
        
        return max_accuracy



    def enable_admin_mode(self):
        """Prompt for password and enable admin mode if correct."""
        global admin_mode
        password = tk.simpledialog.askstring("Admin Login", "Enter admin password:", show="*")
        if password == PASSWORD:
            admin_mode = True
            self.update_admin_mode_display()
            messagebox.showinfo("Admin Mode", "Admin mode enabled.")

            # Remove the Tuning tab after admin access is granted
            if hasattr(self, 'tuning_tab'):
                self.notebook.forget(self.tuning_tab)
                del self.tuning_tab  # Ensure reference is cleared
        else:
            messagebox.showerror("Access Denied", "Incorrect password.")

    def update_admin_mode_display(self):
        """Update the UI based on admin mode status."""
        if admin_mode:
            # Remove the Calibration tab if it exists
            if hasattr(self, 'calibration_tab'):
                self.notebook.forget(self.calibration_tab)
                del self.calibration_tab  # Ensure the reference is cleared

            # Add admin-specific tabs if they don't exist
            if not hasattr(self, 'threshold_tab'):
                self.setup_threshold_settings()
            if not hasattr(self, 'roi_management_tab'):
                self.setup_roi_management()
            if not hasattr(self, 'threshold_optimizer_tab'):
                self.setup_threshold_optimizer_tab()

            # Enable admin functionalities
            self.enable_all_buttons()

            # Show 'Admin Mode' label if it doesn't already exist
            if not hasattr(self, 'admin_label'):
                self.admin_label = tk.Label(self.master, text="Admin Mode", font=("Arial", 16), fg="red")
                self.admin_label.pack()
        else:
            # Hide or disable any admin functionality
            if hasattr(self, 'admin_label'):
                self.admin_label.pack_forget()
            self.disable_all_buttons_except_trigger()

            # Re-add the Calibration tab if it was removed
            if not hasattr(self, 'calibration_tab'):
                self.setup_calibration_tab()

    def create_tuning_tab(self):
        """Create a Tuning tab that prompts for a password to access admin mode."""
        self.tuning_tab = tk.Frame(self.notebook)
        self.notebook.add(self.tuning_tab, text="Tuning")

        # Add a label and button to trigger admin mode password prompt
        tk.Label(self.tuning_tab, text="Admin Access Required", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.tuning_tab, text="Enter Password for Admin Access", command=self.enable_admin_mode).pack(pady=10)

    def enable_admin_mode(self):
        """Prompt for password and enable admin mode if correct."""
        global admin_mode
        password = tk.simpledialog.askstring("Admin Login", "Enter admin password:", show="*")
        if password == PASSWORD:
            admin_mode = True
            self.update_admin_mode_display()
            messagebox.showinfo("Admin Mode", "Admin mode enabled.")
            
            # Remove the Tuning tab after admin access is granted
            self.notebook.forget(self.tuning_tab)
        else:
            messagebox.showerror("Access Denied", "Incorrect password.")

    def update_admin_mode_display(self):
        """Update the UI based on admin mode status."""
        if admin_mode:
            # Display additional tabs and optimizer functionality for admin mode
            self.setup_threshold_settings()
            self.setup_roi_management()
            self.setup_threshold_optimizer_tab()

            # Enable all functionalities in Dashboard (e.g., buttons)
            self.enable_all_buttons()

            # Show 'Admin Mode' label if it doesn't already exist
            if not hasattr(self, 'admin_label'):
                self.admin_label = tk.Label(self.master, text="Admin Mode", font=("Arial", 16), fg="red")
                self.admin_label.pack()
        else:
            # Hide or disable any admin functionality
            if hasattr(self, 'admin_label'):
                self.admin_label.pack_forget()
            self.disable_all_buttons_except_trigger()

    def enable_all_buttons(self):
        """Enable all buttons for admin functionalities."""
        # Iterate through dictionaries for ROI-specific buttons
        for button_dict in [self.rename_buttons, self.mark_buttons, self.capture_buttons, self.reset_buttons, self.set_ref_buttons]:
            for button in button_dict.values():
                button.config(state="normal")

        # Enable other buttons explicitly
        if hasattr(self, 'save_threshold_button'):
            self.save_threshold_button.config(state="normal")
        if hasattr(self, 'save_rois_button'):
            self.save_rois_button.config(state="normal")

    def disable_all_buttons_except_trigger(self):
        """Disable all buttons except manual trigger for normal mode."""
        # Ensure self.trigger_button is initialized
        if not hasattr(self, 'trigger_button'):
            return  # Exit if the trigger button is not created yet

        # Check if buttons exist before attempting to disable them
        buttons = [
            getattr(self, 'save_threshold_button', None),
            getattr(self, 'save_rois_button', None),
            getattr(self, 'rename_button', None),
            getattr(self, 'mark_button', None),
            getattr(self, 'capture_button', None),
            getattr(self, 'reset_button', None)
        ]
        for button in buttons:
            if button:
                button.config(state="disabled")

        if self.trigger_button:
            self.trigger_button.config(state="normal")

    def setup_calibration_tab(self):
        """Set up the Calibration tab with center-aligned content."""
        self.calibration_tab = tk.Frame(self.notebook)
        self.notebook.add(self.calibration_tab, text="Calibration")

        # Use grid to center-align content
        self.calibration_tab.grid_rowconfigure(0, weight=1)
        self.calibration_tab.grid_rowconfigure(1, weight=1)
        self.calibration_tab.grid_rowconfigure(2, weight=1)
        self.calibration_tab.grid_columnconfigure(0, weight=1)

        # Create a container frame for all elements
        container_frame = tk.Frame(self.calibration_tab)
        container_frame.grid(row=0, column=0, sticky="nsew")  # Center-align
        container_frame.grid_rowconfigure(0, weight=1)
        container_frame.grid_rowconfigure(1, weight=1)
        container_frame.grid_rowconfigure(2, weight=1)
        container_frame.grid_columnconfigure(0, weight=1)

        # ROI selection dropdown
        tk.Label(
            container_frame, text="Select ROI:", font=("Arial", 14, "bold")
        ).grid(row=0, column=0, sticky="e", padx=10, pady=10)
        self.calibration_roi_dropdown = ttk.Combobox(
            container_frame,
            values=list(roi_coordinates.keys()),
            state="readonly",
            font=("Arial", 12),
            width=20,
        )
        self.calibration_roi_dropdown.set("Select ROI")
        self.calibration_roi_dropdown.grid(row=0, column=1, sticky="w", padx=10, pady=10)

        # Start calibration button
        self.start_calibration_button = tk.Button(
            container_frame,
            text="Start Calibration",
            command=self.start_calibration,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",  # Green background
            fg="black",  # Black text color
            relief="raised",
            bd=2,
            padx=10,
            pady=5,
        )
        self.start_calibration_button.grid(row=1, column=0, columnspan=2, pady=20)

        # Calibration results section
        tk.Label(
            container_frame, text="Calibration Results:", font=("Arial", 14, "bold")
        ).grid(row=2, column=0, columnspan=2, pady=10)
        self.calibration_results_text = tk.Text(
            container_frame,
            font=("Arial", 12),
            width=60,
            height=8,
            wrap="word",
            state="disabled",
            relief="solid",
            bd=1,
        )
        self.calibration_results_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Recommendations section
        tk.Label(
            container_frame, text="Recommendations:", font=("Arial", 14, "bold")
        ).grid(row=4, column=0, columnspan=2, pady=10)
        self.recommendations_text = tk.Text(
            container_frame,
            font=("Arial", 12),
            width=60,
            height=4,
            wrap="word",
            state="disabled",
            relief="solid",
            bd=1,
        )
        self.recommendations_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10)


    def start_calibration(self):
        """Perform calibration by triggering the system 10 times and analyzing confidence."""
        selected_roi = self.calibration_roi_dropdown.get()
        if not selected_roi or selected_roi not in roi_coordinates:
            messagebox.showerror("Error", "Please select a valid ROI.")
            return

        reference_images = self.load_reference_images(ROI_FOLDERS[selected_roi])
        if not reference_images:
            self.update_calibration_log(f"ROI {selected_roi}: No valid reference images found.")
            return

        # Get ROI coordinates
        roi_coords = roi_coordinates[selected_roi]
        if not roi_coords or any(v == 0 for v in roi_coords.values()):
            self.update_calibration_log(f"ROI {selected_roi}: Invalid or missing ROI coordinates.")
            return

        calibration_results = []
        max_confidences = []

        # Prompt user to trigger 10 times
        for i in range(10):
            messagebox.showinfo("Trigger", f"Please perform assembly trigger {i + 1}/10.")
            ret, frame = cap.read()
            if not ret:
                self.update_calibration_log(f"Assembly {i + 1}: Unable to capture frame.")
                continue

            try:
                # Crop the frame to the selected ROI
                frame_cropped = frame[roi_coords['y1']:roi_coords['y2'], roi_coords['x1']:roi_coords['x2']]
                frame_gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)

                # Perform template matching with all reference images
                confidences = []
                for ref_image in reference_images:
                    if ref_image.ndim > 2:  # Ensure reference image is grayscale
                        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

                    # Ensure dimensions match
                    if frame_gray.shape != ref_image.shape:
                        raise ValueError(f"Image dimensions do not match. Frame: {frame_gray.shape}, Template: {ref_image.shape}")

                    res = self.template_matching(frame_gray, ref_image)
                    confidences.append(res.max())

                max_confidence = max(confidences) if confidences else 0
                max_confidences.append(max_confidence)
                self.update_calibration_log(f"Assembly {i + 1}: Confidence = {max_confidence:.2f}")
            except Exception as e:
                self.update_calibration_log(f"Assembly {i + 1}: Error during template matching: {e}")
                continue

        # Calculate average confidence
        if max_confidences:
            avg_confidence = sum(max_confidences) / len(max_confidences)
            calibration_results.append(f"ROI {selected_roi}: Avg Confidence = {avg_confidence:.2f}")

            # Provide recommendations based on average confidence
            recommendations = []
            if avg_confidence < 0.55:
                recommendations.append(f"ROI {selected_roi}: Consider capturing new reference images.")
            else:
                recommendations.append(f"ROI {selected_roi}: Threshold adjustment may be sufficient.")

            # Update the UI with results
            self.calibration_results_text.config(state="normal")
            self.calibration_results_text.delete(1.0, "end")
            self.calibration_results_text.insert("end", "\n".join(calibration_results))
            self.calibration_results_text.config(state="disabled")

            self.recommendations_text.config(state="normal")
            self.recommendations_text.delete(1.0, "end")
            self.recommendations_text.insert("end", "\n".join(recommendations))
            self.recommendations_text.config(state="disabled")

            # Save calibration date after successful calibration
            self.update_calibration_date()

        else:
            self.update_calibration_log(f"ROI {selected_roi}: Insufficient data for analysis.")




    def update_calibration_log(self, message):
        """Update the calibration log with individual results."""
        self.calibration_results_text.config(state="normal")
        self.calibration_results_text.insert("end", message + "\n")
        self.calibration_results_text.see("end")  # Scroll to the latest entry
        self.calibration_results_text.config(state="disabled")



    def setup_threshold_optimizer_tab(self):
        """Set up the Threshold Optimizer tab for OK and NOT OK optimization."""
        self.threshold_optimizer_tab = tk.Frame(self.notebook)
        self.notebook.add(self.threshold_optimizer_tab, text="Threshold Optimizer")

        # Center-align content
        self.threshold_optimizer_tab.grid_columnconfigure(0, weight=1)
        self.threshold_optimizer_tab.grid_columnconfigure(1, weight=1)
        self.threshold_optimizer_tab.grid_rowconfigure(0, weight=1)

        # ROI Selection
        tk.Label(
            self.threshold_optimizer_tab,
            text="Select ROI:",
            font=("Arial", 14)
        ).grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.selected_roi = ttk.Combobox(
            self.threshold_optimizer_tab,
            values=list(roi_coordinates.keys()),  # Populate with ROI options
            font=("Arial", 14),
            state="readonly",
            width=20
        )
        self.selected_roi.set("Select ROI")
        self.selected_roi.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # OK Optimizer Button
        self.ok_optimizer_button = tk.Button(
            self.threshold_optimizer_tab,
            text="Run OK Optimizer",
            command=self.start_ok_optimizer,
            font=("Arial", 14),
            bg="#4CAF50",
            fg="black",
            width=20
        )
        self.ok_optimizer_button.grid(row=1, column=0, columnspan=2, pady=10)

        # NOT OK Finder Button
        self.not_ok_finder_button = tk.Button(
            self.threshold_optimizer_tab,
            text="Run NOT OK Finder",
            command=self.start_not_ok_finder,
            font=("Arial", 14),
            bg="#FF5733",
            fg="black",
            width=20
        )
        self.not_ok_finder_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Result Display Box
        tk.Label(
            self.threshold_optimizer_tab,
            text="Optimized Threshold:",
            font=("Arial", 14)
        ).grid(row=3, column=0, padx=10, pady=5, sticky="e")

        self.optimized_threshold_display = tk.Entry(
            self.threshold_optimizer_tab,
            font=("Arial", 14),
            justify="center",
            state="readonly",
            width=10
        )
        self.optimized_threshold_display.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    def start_ok_optimizer(self):
        """Run the OK optimizer for the selected ROI."""
        roi = self.selected_roi.get()
        if roi == "Select ROI":
            messagebox.showerror("Error", "Please select a valid ROI.")
            return

        # Initial threshold values and variables
        threshold_values = np.arange(0.1, 1.1, 0.1)
        optimal_threshold = None

        for threshold in threshold_values:
            thresholds[roi] = {'Threshold': threshold, 'Display Name': roi}
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Unable to access camera.")
                return

            result = self.template_match(frame, roi)
            if result != "OK":
                optimal_threshold = threshold - 0.1
                break

        # Update display box with the result
        if optimal_threshold:
            thresholds[roi]['Threshold'] = optimal_threshold
            self.optimized_threshold_display.config(state="normal")
            self.optimized_threshold_display.delete(0, tk.END)
            self.optimized_threshold_display.insert(0, f"{optimal_threshold:.2f}")
            self.optimized_threshold_display.config(state="readonly")
        else:
            self.optimized_threshold_display.config(state="normal")
            self.optimized_threshold_display.delete(0, tk.END)
            self.optimized_threshold_display.insert(0, "N/A")
            self.optimized_threshold_display.config(state="readonly")

    def start_not_ok_finder(self):
        """Run the NOT OK optimizer for the selected ROI."""
        roi = self.selected_roi.get()
        if roi == "Select ROI":
            messagebox.showerror("Error", "Please select a valid ROI.")
            return

        # Fixed starting threshold for decremental testing
        starting_threshold = 0.9

        # Save the original threshold to revert after the run
        original_threshold = thresholds.get(roi, {}).get('Threshold', 0.7)

        # Threshold values in decremental order from the starting threshold
        threshold_values = np.arange(starting_threshold, 0, -0.1)

        # Capture a frame for testing (frame should be in "NOT OK" condition for this finder)
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Unable to access camera.")
            return

        # Ensure the current frame is set to "NOT OK"
        initial_result = self.template_match(frame, roi)
        if initial_result != "NOT OK":
            messagebox.showerror("Error", "Frame is not in a NOT OK condition. Please adjust the frame and try again.")
            return

        # Debug: Check initial conditions
        print(f"Initial frame condition: {initial_result} (Expected: 'NOT OK')")

        optimal_threshold = None  # To track the first transition from NOT OK to OK
        last_threshold_tested = None  # To track the last tested threshold

        for threshold in threshold_values:
            # Temporarily set the threshold for testing
            thresholds[roi] = {'Threshold': threshold, 'Display Name': roi}
            result = self.template_match(frame, roi)

            # Debug: Log threshold and result
            print(f"Testing Threshold: {threshold:.2f}, Result: {result}")
            last_threshold_tested = threshold

            if result == "OK":
                # Stop at the first threshold where NOT OK transitions to OK
                optimal_threshold = threshold
                break

        # If no optimal threshold was found, display the last threshold tested
        if optimal_threshold is None:
            optimal_threshold = last_threshold_tested

        # Revert the threshold to the original value
        thresholds[roi]['Threshold'] = original_threshold
        save_thresholds()  # Save back the original threshold to the CSV

        # Update the result display box with the threshold
        self.optimized_threshold_display.config(state="normal")
        self.optimized_threshold_display.delete(0, tk.END)
        self.optimized_threshold_display.insert(0, f"{optimal_threshold:.2f}")
        self.optimized_threshold_display.config(state="readonly")

        # Notify completion
        messagebox.showinfo(
            "NOT OK Finder",
            f"Run completed. Optimal NOT OK Threshold: {optimal_threshold:.2f} (Last tested: {last_threshold_tested:.2f})"
        )

    def optimize_ok_threshold(self):
        """Optimize OK threshold for the selected ROI using incremental steps."""
        selected_roi = self.prompt_select_roi()
        if not selected_roi:
            return  # Exit if no valid ROI is selected

        # Ensure selected ROI exists in thresholds with a default threshold if missing
        if selected_roi not in thresholds:
            thresholds[selected_roi] = {'Threshold': 0.7, 'Display Name': selected_roi}

        threshold_values = np.arange(0.1, 1.1, 0.1)
        optimal_threshold = 0.1  # Initialize with the lowest threshold
        original_threshold = thresholds[selected_roi]['Threshold']

        # Temporarily update threshold for testing
        for threshold in threshold_values:
            thresholds[selected_roi]['Threshold'] = threshold
            logging.info(f"Testing threshold: {threshold} for ROI: {selected_roi}")

            # Use the template_match function for the actual test
            ret, frame = cap.read()  # Get current frame for testing
            if not ret:
                messagebox.showerror("Error", "Unable to capture frame for OK optimization.")
                return

            result = self.template_match(frame, selected_roi)
            is_ok = result == "OK"
            self.display_threshold_result(threshold, is_ok)

            if not is_ok:
                optimal_threshold = threshold - 0.1  # Set the last OK threshold as optimal
                break

        # Display final result without saving it
        self.result_label.config(text=f"Optimized OK Threshold for {selected_roi}: {optimal_threshold}")
        thresholds[selected_roi]['Threshold'] = original_threshold  # Restore original threshold

    def display_threshold_result(self, threshold, is_ok):
        """Display each threshold test result on the screen and log for debugging."""
        status_text = "OK" if is_ok else "NOT OK"
        self.result_label.config(text=f"Threshold: {threshold:.1f} - Result: {status_text}")
        logging.info(f"Threshold: {threshold:.1f} - Result: {status_text}")
        self.master.update_idletasks()
        self.master.after(1000)  # Brief pause for visibility in UI

    def load_reference_images(self, folder_path):
        """Loads and returns reference images from a given folder."""
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        return images
    
    def template_matching(self, image, template):
        """Enhanced template matching with multiple methods"""
        processed_image = self.preprocess_image(image)
        processed_template = self.preprocess_image(template)
        
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        max_result = None
        
        for method in methods:
            result = cv2.matchTemplate(processed_image, processed_template, method)
            if max_result is None or np.max(result) > np.max(max_result):
                max_result = result
                
        return max_result

    def plot_threshold_metrics(self, metrics, title):
        thresholds, rates = zip(*metrics)
        plt.plot(thresholds, rates, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title(title)
        plt.grid(True)
        plt.show()

    def preprocess_image(self, image):
        """Enhance image features for better O-ring detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        binary = cv2.adaptiveThreshold(enhanced, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        return binary
    
    def validate_oring(self, binary_image):
        """Validate O-ring presence using contour analysis"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4*np.pi*area/(perimeter*perimeter) if perimeter > 0 else 0
            
            # Adjust these thresholds based on your O-ring size
            if 0.7 < circularity < 1.0 and 100 < area < 5000:
                return True
        return False

    def template_match(self, frame, roi):
        try:
            # Get paths for both OK and NOT_OK references
            ok_folder = ROI_FOLDERS.get(roi, "")
            not_ok_folder = os.path.join(ok_folder, "NOT_OK")

            coords = roi_coordinates[roi]
            cropped = frame[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
            processed_frame = self.preprocess_image(cropped)

            # Get max accuracy for OK references
            ok_accuracy = 0
            ok_templates = [cv2.imread(f) for f in glob.glob(f"{ok_folder}/*.jpg") if os.path.isfile(f)]
            for template in ok_templates:
                if template is not None and cropped.size > 0:
                    processed_template = self.preprocess_image(template)
                    
                    # Use multiple matching methods
                    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                    for method in methods:
                        res = cv2.matchTemplate(processed_frame, processed_template, method)
                        _, confidence, _, _ = cv2.minMaxLoc(res)
                        ok_accuracy = max(ok_accuracy, confidence)

            # Get max accuracy for NOT_OK references
            not_ok_accuracy = 0
            not_ok_templates = [cv2.imread(f) for f in glob.glob(f"{not_ok_folder}/*.jpg") if os.path.isfile(f)]
            for template in not_ok_templates:
                if template is not None and cropped.size > 0:
                    processed_template = self.preprocess_image(template)
                    
                    for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
                        res = cv2.matchTemplate(processed_frame, processed_template, method)
                        _, confidence, _, _ = cv2.minMaxLoc(res)
                        not_ok_accuracy = max(not_ok_accuracy, confidence)

            # Validate O-ring presence using contour analysis
            contour_validation = self.validate_oring(processed_frame)

            # Combined decision logic
            if ok_accuracy > not_ok_accuracy and contour_validation:
                roi_color[roi] = (0, 255, 0)
                roi_status[roi] = "OK"
                return "OK"
            else:
                roi_color[roi] = (0, 0, 255)
                roi_status[roi] = "NOT OK"
                return "NOT OK"

        except Exception as e:
            logging.error(f"Template matching error for {roi}: {str(e)}")
            return "NOT OK"



    def reset_rois(self):
        for roi in roi_coordinates.keys():
            roi_color[roi] = (255, 0, 0)
            roi_status[roi] = "Waiting"
        self.assembly_status = ""

    def reset_roi_references(self, roi):
        """Reset reference images with a confirmation prompt."""
        folder = ROI_FOLDERS.get(roi, f"{roi}_Images")
        if folder and os.path.exists(folder) and os.listdir(folder):
            confirm = messagebox.askyesno(
                "Warning", 
                f"Resetting the reference images will permanently delete all existing images for {roi}!\n\n"
                f"Are you sure you want to continue?"
            )
            if not confirm:
                return  # Exit if user cancels

        # Delete all images inside the ROI folder
        for f in glob.glob(f"{folder}/*.jpg"):
            os.remove(f)

        messagebox.showinfo("Success", f"All reference images for {roi} have been cleared.")

    def start_camera_feed(self):
        """Initialize the camera feed to avoid duplicate feeds."""
        global cap
        cap = cv2.VideoCapture(0)
        self.update_feed()

    # def start_camera_feed(self):
    #     """Initialize the video feed from MP4 file."""
    #     global cap
    #     video_path = "./h6.mp4"  # Replace with your video file path
    #     cap = cv2.VideoCapture(video_path)
        
    #     # Set video to loop when it reaches the end
    #     def check_video_end():
    #         if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #         self.master.after(1, check_video_end)
        
    #     check_video_end()
    #     self.update_feed()


    def exit_fullscreen(self, event=None):
        """Exit full-screen mode when Escape is pressed."""
        self.master.attributes('-fullscreen', False)

    def update_feed(self):
        """Apply zoom and focus settings to the video feed and draw ROIs."""
        global cap

        ret, frame = cap.read()
        if ret:
            # Retrieve the current zoom and focus settings
            roi = next(iter(thresholds.keys()))  # Example: use the first ROI's settings
            zoom = thresholds[roi].get('Zoom', 1.0)
            focus = thresholds[roi].get('Focus', 1.0)

            # Apply focus
            cap.set(cv2.CAP_PROP_FOCUS, focus)

            # Apply zoom
            height, width = frame.shape[:2]
            new_width, new_height = int(width / zoom), int(height / zoom)
            x_start = (width - new_width) // 2
            y_start = (height - new_height) // 2
            cropped_frame = frame[y_start:y_start + new_height, x_start:x_start + new_width]
            resized_frame = cv2.resize(cropped_frame, (640, 420))

            # Draw ROIs
            for roi_name, coords in roi_coordinates.items():
                # Scale the coordinates to match the resized frame
                scale_x = resized_frame.shape[1] / new_width
                scale_y = resized_frame.shape[0] / new_height
                x1 = int((coords['x1'] - x_start) * scale_x)
                y1 = int((coords['y1'] - y_start) * scale_y)
                x2 = int((coords['x2'] - x_start) * scale_x)
                y2 = int((coords['y2'] - y_start) * scale_y)

                # Draw rectangle and label
                if 0 <= x1 < resized_frame.shape[1] and 0 <= y1 < resized_frame.shape[0] and \
                   0 <= x2 < resized_frame.shape[1] and 0 <= y2 < resized_frame.shape[0]:
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), roi_color.get(roi_name, (255, 0, 0)), 2)
                    cv2.putText(
                        resized_frame, 
                        coords.get('Display Name', roi_name), 
                        (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )

            # Draw assembly status
            text_size = cv2.getTextSize(self.assembly_status, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (resized_frame.shape[1] - text_size[0]) // 2
            text_y = (resized_frame.shape[0] + text_size[1]) // 2
            cv2.putText(
                resized_frame, 
                self.assembly_status, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                self.status_color, 
                3
            )

            # Convert to Tkinter-compatible format
            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)

        self.master.after(10, self.update_feed)

    # def update_feed(self):
    #     """Apply zoom and focus settings to the video feed and draw ROIs."""
    #     global cap

    #     ret, frame = cap.read()
    #     if ret:
    #         # Get zoom/focus settings from first ROI
    #         roi = next(iter(thresholds.keys()))
    #         zoom = thresholds[roi].get('Zoom', 1.0)
    #         focus = thresholds[roi].get('Focus', 1.0)

    #         # Apply focus
    #         cap.set(cv2.CAP_PROP_FOCUS, focus)

    #         # Apply zoom and get display frame
    #         height, width = frame.shape[:2]
    #         new_width, new_height = int(width / zoom), int(height / zoom)
    #         x_start = (width - new_width) // 2
    #         y_start = (height - new_height) // 2
    #         cropped_frame = frame[y_start:y_start + new_height, x_start:x_start + new_width]
    #         display_frame = cv2.resize(cropped_frame, (640, 420))

    #         # Draw ROIs
    #         for roi_name, coords in roi_coordinates.items():
    #             # Use direct ROI coordinates
    #             x1, y1 = coords['x1'], coords['y1']
    #             x2, y2 = coords['x2'], coords['y2']

    #             # Draw rectangle and label
    #             cv2.rectangle(display_frame, (x1, y1), (x2, y2), roi_color.get(roi_name, (255, 0, 0)), 2)
    #             cv2.putText(
    #                 display_frame, 
    #                 coords.get('Display Name', roi_name), 
    #                 (x1, max(0, y1 - 10)), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.5, 
    #                 (255, 255, 255), 
    #                 1
    #             )

    #         # Draw assembly status
    #         text_size = cv2.getTextSize(self.assembly_status, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    #         text_x = (display_frame.shape[1] - text_size[0]) // 2
    #         text_y = (display_frame.shape[0] + text_size[1]) // 2
    #         cv2.putText(
    #             display_frame, 
    #             self.assembly_status, 
    #             (text_x, text_y), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 
    #             1.5, 
    #             self.status_color, 
    #             3
    #         )

    #         # Convert and display
    #         img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    #         imgtk = ImageTk.PhotoImage(image=img)
    #         self.camera_label.imgtk = imgtk
    #         self.camera_label.config(image=imgtk)

    #     self.master.after(self.frame_delay, self.update_feed)





    def on_close(self):
        # if self.serial_conn and self.serial_conn.is_open:
        #     self.serial_conn.close()
        # if cap and cap.isOpened():
        #     cap.release()
        # self.master.destroy()
        if plc.IS_CONNECTED and self.plc_conn:
            self.plc.disconnect()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        self.master.quit()
        self.master.destroy()
        
    def start_heartbeat(self):
        if plc.IS_CONNECTED and self.plc_conn and hasattr(self.plc, "send_heartbeat"):
            success = self.plc.send_heartbeat()
            if not success:
                print(f"PLC heartbeat failed - attempting reconnection")
                self.plc_conn = False
                self.update_plc_status_display()
                if not self.reconnecting:
                    self.reconnecting = True
                    threading.Thread(target=self.reconnect_plc, daemon=True).start()
        self.master.after(5000, self.start_heartbeat)  # Schedule next heartbeat in 5 seconds

    def reconnect_plc(self):
        print("Starting PLC reconnection attempts...")
        while True:
            success = self.plc.connect()
            if success:
                self.plc_conn = True
                self.update_plc_status_display()
                self.reconnecting = False
                print("PLC reconnected successfully")
                break
            time.sleep(1)
def main():
    root = tk.Tk()
    app = VisionInspectionUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()  # Start the main application