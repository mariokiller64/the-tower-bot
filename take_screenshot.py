# take_screenshot.py - Updated version
import os
import argparse
from datetime import datetime
from pathlib import Path
from android_device import AndroidDevice

def main():
    parser = argparse.ArgumentParser(description="Capture screenshot from Android device")
    parser.add_argument('--device', type=str, default='auto', help='Device ID')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--dir', type=str, default='screenshots', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.dir)
    output_dir.mkdir(exist_ok=True)
    
    # Connect to device
    try:
        device = AndroidDevice(args.device)
    except Exception as e:
        print(f"Failed to connect to device: {e}")
        return 1
    
    # Capture screenshot
    cap = device.capture()
    if cap is None:
        print('Capture failed')
        return 1
    
    # Generate filename
    if args.output:
        filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
    
    # Save image
    output_path = output_dir / filename
    
    # Convert numpy array to PIL Image and save
    from PIL import Image
    img = Image.fromarray(cap)
    img.save(output_path)
    
    print(f"Screenshot saved to: {output_path}")
    return 0

if __name__ == "__main__":
    exit(main())
