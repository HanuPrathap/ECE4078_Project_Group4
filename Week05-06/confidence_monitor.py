#!/usr/bin/env python3
"""
Simple Confidence Monitor
Tracks target confidence values and prints to terminal
Exits when confidence threshold is met
"""

import json
import os
import time
import sys
from collections import defaultdict

class ConfidenceMonitor:
    def __init__(self, confidence_threshold=0.8, max_detections=5):
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.target_data = defaultdict(list)
        self.monitoring = True
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        """Print header information"""
        print("🎯 LIVE CONFIDENCE MONITOR")
        print("=" * 50)
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Max Detections: {self.max_detections}")
        print("=" * 50)
        print()
        
    def check_files(self):
        """Check for target data files"""
        files_to_check = [
            'lab_output/live_targets.txt',
            'lab_output/targets.txt'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    return data
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
        return None
        
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
                
                # Keep only last 50 entries per target
                if len(self.target_data[target_name]) > 50:
                    self.target_data[target_name] = self.target_data[target_name][-50:]
    
    def print_current_status(self):
        """Print current confidence status"""
        if not self.target_data:
            print("⏳ Waiting for target detections...")
            return False
            
        print("📊 CURRENT TARGET CONFIDENCE STATUS")
        print("-" * 40)
        
        all_targets_ready = True
        
        for target_name, history in self.target_data.items():
            if not history:
                continue
                
            latest = history[-1]
            confidence = latest['confidence']
            uncertainty = latest['uncertainty']
            n_detections = latest['n_detections']
            
            # Check if this target meets criteria
            target_ready = (confidence >= self.confidence_threshold and 
                           n_detections >= self.max_detections)
            
            if not target_ready:
                all_targets_ready = False
            
            # Status indicators
            if confidence >= 0.8:
                status = "🟢 HIGH"
            elif confidence >= 0.6:
                status = "🟡 MEDIUM"
            else:
                status = "🔴 LOW"
            
            ready_indicator = "✅ READY" if target_ready else "⏳ PENDING"
            
            print(f"{target_name}:")
            print(f"  Confidence: {confidence:.3f} {status}")
            print(f"  Uncertainty: {uncertainty:.3f} m")
            print(f"  Detections: {n_detections}/{self.max_detections}")
            print(f"  Status: {ready_indicator}")
            print()
        
        return all_targets_ready
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 50)
        print("🎯 FINAL SUMMARY")
        print("=" * 50)
        
        total_targets = len(self.target_data)
        ready_targets = 0
        
        for target_name, history in self.target_data.items():
            if not history:
                continue
                
            latest = history[-1]
            confidence = latest['confidence']
            n_detections = latest['n_detections']
            
            if (confidence >= self.confidence_threshold and 
                n_detections >= self.max_detections):
                ready_targets += 1
                
            print(f"{target_name}: {confidence:.3f} ({n_detections} detections)")
        
        print(f"\nReady targets: {ready_targets}/{total_targets}")
        
        if ready_targets == total_targets and total_targets > 0:
            print("🎉 ALL TARGETS READY!")
            return True
        else:
            print("⏳ Still waiting for targets...")
            return False
    
    def monitor(self):
        """Main monitoring loop"""
        self.clear_screen()
        self.print_header()
        
        iteration = 0
        
        while self.monitoring:
            iteration += 1
            
            # Check for new data
            data = self.check_files()
            if data:
                self.update_target_data(data)
            
            # Print current status
            print(f"🔄 Iteration {iteration} - {time.strftime('%H:%M:%S')}")
            print()
            
            all_ready = self.print_current_status()
            
            if all_ready and len(self.target_data) > 0:
                print("\n🎉 SUCCESS! All targets meet criteria!")
                self.print_summary()
                break
            
            print(f"\n⏳ Polling every 2 seconds... (Ctrl+C to stop)")
            print("-" * 50)
            
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                print("\n\n🛑 Monitoring stopped by user")
                self.print_summary()
                break

def main():
    """Main function"""
    print("Starting Confidence Monitor...")
    print("This will track target confidence values")
    print("Press Ctrl+C to stop\n")
    
    # Get user preferences
    try:
        threshold = float(input("Enter confidence threshold (0.0-1.0) [default: 0.8]: ") or "0.8")
        max_detections = int(input("Enter max detections required [default: 5]: ") or "5")
    except ValueError:
        threshold = 0.8
        max_detections = 5
    
    # Create and start monitor
    monitor = ConfidenceMonitor(confidence_threshold=threshold, max_detections=max_detections)
    monitor.monitor()

if __name__ == "__main__":
    main()
