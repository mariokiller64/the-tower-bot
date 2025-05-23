# run_state_check.py - Updated state checking tool
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from android_device import AndroidDevice
from state import create_state_manager

def main():
    parser = argparse.ArgumentParser(description="Check game state detection")
    parser.add_argument('--device', type=str, default='auto', help='Device ID')
    parser.add_argument('--interval', type=float, default=3.0, help='Check interval (seconds)')
    parser.add_argument('--save-captures', action='store_true', help='Save captures for debugging')
    
    args = parser.parse_args()
    
    # Setup
    device = AndroidDevice(args.device)
    state_manager = create_state_manager()
    
    if args.save_captures:
        debug_dir = Path("debug_captures")
        debug_dir.mkdir(exist_ok=True)
    
    print("State detection checker running. Press Ctrl+C to stop.")
    print("-" * 50)
    
    try:
        while True:
            # Capture screen
            cap = device.capture()
            if cap is None:
                print('Capture failed')
                time.sleep(args.interval)
                continue
            
            # Detect state
            state = state_manager.detect_state(cap)
            
            # Get confidence scores
            if hasattr(state_manager, 'confidence_history') and state_manager.confidence_history:
                latest = state_manager.confidence_history[-1]
                all_scores = latest.get('all_confidences', {})
                
                # Display results
                print(f"\nTimestamp: {time.strftime('%H:%M:%S')}")
                if state:
                    print(f"Detected State: {state.name}")
                else:
                    print("Detected State: UNKNOWN")
                
                print("Confidence Scores:")
                for state_name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(score * 20)
                    print(f"  {state_name:<20} {score:.2%} {bar}")
                
                # Save debug capture if requested
                if args.save_captures and (not state or latest.get('confidence', 0) < 0.9):
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    filename = f"debug_{timestamp}_{state.name if state else 'UNKNOWN'}.png"
                    cv2.imwrite(str(debug_dir / filename), cv2.cvtColor(cap, cv2.COLOR_RGB2BGR))
                    print(f"  Saved debug capture: {filename}")
            
            print("-" * 50)
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nStopped by user")

if __name__ == "__main__":
    main()
