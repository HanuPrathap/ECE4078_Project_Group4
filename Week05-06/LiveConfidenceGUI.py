# Live Confidence GUI for Target Pose Estimation
# Displays real-time confidence values for detected targets

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import os
import time
import threading
from collections import defaultdict

class LiveConfidenceGUI:
    """GUI for displaying live confidence values of detected targets"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Target Confidence Monitor")
        self.root.geometry("1000x700")
        
        # Data storage
        self.target_data = defaultdict(list)  # Store confidence history for each target
        self.current_detections = {}  # Current detection data
        self.monitoring = True
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎯 Live Target Confidence Monitor", 
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Control buttons
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", 
                                     command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Monitoring", 
                                    command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(control_frame, text="Clear Data", 
                                     command=self.clear_data)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_button = ttk.Button(control_frame, text="Save Data", 
                                    command=self.save_data)
        self.save_button.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Monitoring: OFF")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack()
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_var = tk.StringVar(value="Total detections: 0 | Active targets: 0")
        self.stats_label = ttk.Label(stats_frame, textvariable=self.stats_var)
        self.stats_label.pack()
        
        # Confidence display frame
        confidence_frame = ttk.LabelFrame(main_frame, text="Live Confidence Values", padding=10)
        confidence_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(confidence_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Individual target tabs
        self.target_tabs = {}
        self.target_texts = {}
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, height=15, width=80, 
                                                    font=("Consolas", 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_files, daemon=True)
        self.monitor_thread.start()
        
    def start_monitoring(self):
        """Start monitoring for target data"""
        self.monitoring = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Monitoring: ON - Watching for target data...")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Monitoring: OFF")
        
    def clear_data(self):
        """Clear all stored data"""
        self.target_data.clear()
        self.current_detections.clear()
        
        # Clear all text widgets
        self.summary_text.delete(1.0, tk.END)
        for text_widget in self.target_texts.values():
            text_widget.delete(1.0, tk.END)
        
        # Remove all target tabs except summary
        for target_name in list(self.target_tabs.keys()):
            self.notebook.forget(self.target_tabs[target_name])
            del self.target_tabs[target_name]
            del self.target_texts[target_name]
        
        self.update_stats()
        
    def save_data(self):
        """Save confidence data to file"""
        if not self.target_data:
            self.status_var.set("No data to save")
            return
            
        filename = f"confidence_data_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(dict(self.target_data), f, indent=4)
        self.status_var.set(f"Data saved to {filename}")
        
    def monitor_files(self):
        """Monitor for target data files"""
        while True:
            if self.monitoring:
                # Check for live targets file (from TargetPoseEst01.py)
                if os.path.exists('lab_output/live_targets.txt'):
                    try:
                        with open('lab_output/live_targets.txt', 'r') as f:
                            data = json.load(f)
                        self.update_target_data(data)
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass
                
                # Check for main targets file (from TargetPoseEst01.py)
                if os.path.exists('lab_output/targets.txt'):
                    try:
                        with open('lab_output/targets.txt', 'r') as f:
                            data = json.load(f)
                        self.update_target_data(data)
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass
                
                # Check for operate.py image data (for live detection monitoring)
                if os.path.exists('lab_output/images.txt'):
                    try:
                        with open('lab_output/images.txt', 'r') as f:
                            lines = f.readlines()
                        if lines:
                            # Parse the latest image entry
                            latest_entry = json.loads(lines[-1].strip())
                            self.update_operate_data(latest_entry)
                    except (json.JSONDecodeError, FileNotFoundError, IndexError):
                        pass
            
            time.sleep(0.5)  # Check every 500ms
            
    def update_target_data(self, data):
        """Update target data with new detections"""
        current_time = time.time()
        
        for target_name, target_info in data.items():
            if 'confidence' in target_info:
                # Store confidence history
                self.target_data[target_name].append({
                    'timestamp': current_time,
                    'confidence': target_info['confidence'],
                    'x': target_info.get('x', 0),
                    'y': target_info.get('y', 0),
                    'uncertainty': target_info.get('uncertainty', 0),
                    'n_detections': target_info.get('n_detections', 1)
                })
                
                # Keep only last 100 entries per target
                if len(self.target_data[target_name]) > 100:
                    self.target_data[target_name] = self.target_data[target_name][-100:]
                
                # Update current detections
                self.current_detections[target_name] = target_info
        
        # Update GUI
        self.root.after(0, self.update_display)
        
    def update_operate_data(self, image_entry):
        """Update data from operate.py image entries"""
        # This shows that operate.py is actively detecting
        # We can show the robot pose and image info
        if 'pose' in image_entry:
            pose = image_entry['pose']
            self.status_var.set(f"Monitoring: ON - Robot at ({pose[0]:.2f}, {pose[1]:.2f}) - Detection active")
        
        # Update stats to show operate.py is running
        self.update_stats()
        
    def update_display(self):
        """Update the GUI display"""
        self.update_summary()
        self.update_individual_tabs()
        self.update_stats()
        
    def update_summary(self):
        """Update the summary tab"""
        self.summary_text.delete(1.0, tk.END)
        
        if not self.current_detections:
            self.summary_text.insert(tk.END, "No targets detected yet...")
            return
        
        # Create summary
        summary = "🎯 LIVE TARGET CONFIDENCE SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        for target_name, info in self.current_detections.items():
            confidence = info.get('confidence', 0)
            uncertainty = info.get('uncertainty', 0)
            n_detections = info.get('n_detections', 1)
            
            # Confidence color coding
            if confidence >= 0.8:
                conf_status = "🟢 HIGH"
            elif confidence >= 0.6:
                conf_status = "🟡 MEDIUM"
            else:
                conf_status = "🔴 LOW"
            
            summary += f"📊 {target_name.upper()}:\n"
            summary += f"   Confidence: {confidence:.3f} {conf_status}\n"
            summary += f"   Uncertainty: {uncertainty:.3f} m\n"
            summary += f"   Detections: {n_detections}\n"
            summary += f"   Position: ({info.get('x', 0):.3f}, {info.get('y', 0):.3f})\n"
            summary += f"   History: {len(self.target_data[target_name])} entries\n\n"
        
        self.summary_text.insert(tk.END, summary)
        
    def update_individual_tabs(self):
        """Update individual target tabs"""
        for target_name, info in self.current_detections.items():
            if target_name not in self.target_tabs:
                self.create_target_tab(target_name)
            
            # Update target tab content
            text_widget = self.target_texts[target_name]
            text_widget.delete(1.0, tk.END)
            
            # Get confidence history
            history = self.target_data[target_name]
            if not history:
                text_widget.insert(tk.END, f"No data for {target_name}")
                continue
            
            # Create detailed report
            report = f"📈 {target_name.upper()} CONFIDENCE HISTORY\n"
            report += "=" * 40 + "\n\n"
            
            # Current values
            current = history[-1]
            report += f"🔄 CURRENT VALUES:\n"
            report += f"   Confidence: {current['confidence']:.3f}\n"
            report += f"   Uncertainty: {current['uncertainty']:.3f} m\n"
            report += f"   Position: ({current['x']:.3f}, {current['y']:.3f})\n"
            report += f"   Detections: {current['n_detections']}\n"
            report += f"   Timestamp: {time.ctime(current['timestamp'])}\n\n"
            
            # Statistics
            confidences = [h['confidence'] for h in history]
            if len(confidences) > 1:
                report += f"📊 STATISTICS:\n"
                report += f"   Average: {sum(confidences)/len(confidences):.3f}\n"
                report += f"   Min: {min(confidences):.3f}\n"
                report += f"   Max: {max(confidences):.3f}\n"
                report += f"   Trend: {'📈' if confidences[-1] > confidences[0] else '📉'}\n\n"
            
            # Recent history (last 10 entries)
            report += f"📋 RECENT HISTORY (last 10 entries):\n"
            recent = history[-10:]
            for i, entry in enumerate(recent):
                timestamp_str = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                report += f"   {i+1:2d}. {timestamp_str} - Conf: {entry['confidence']:.3f}\n"
            
            text_widget.insert(tk.END, report)
            
    def create_target_tab(self, target_name):
        """Create a new tab for a target"""
        # Create frame for this target
        target_frame = ttk.Frame(self.notebook)
        self.notebook.add(target_frame, text=target_name)
        
        # Create text widget for this target
        target_text = scrolledtext.ScrolledText(target_frame, height=15, width=80, 
                                              font=("Consolas", 10))
        target_text.pack(fill=tk.BOTH, expand=True)
        
        # Store references
        self.target_tabs[target_name] = target_frame
        self.target_texts[target_name] = target_text
        
    def update_stats(self):
        """Update statistics display"""
        total_detections = sum(len(history) for history in self.target_data.values())
        active_targets = len(self.current_detections)
        
        self.stats_var.set(f"Total detections: {total_detections} | Active targets: {active_targets}")
        
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
        
    def close(self):
        """Close the GUI"""
        self.monitoring = False
        self.root.quit()
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    print("🎯 Starting Live Confidence GUI...")
    print("   This GUI will monitor target confidence values in real-time")
    print("   Make sure to run your target pose estimation alongside this GUI")
    print("   Press Ctrl+C to stop\n")
    
    gui = LiveConfidenceGUI()
    gui.run()
