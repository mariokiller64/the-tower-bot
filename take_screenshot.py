import os
from android_device import AndroidDevice
from PIL import Image, ImageDraw

# Find android device
device = AndroidDevice('emulator-5554')

# Capture image
cap = device.capture()
if cap is None:
    print('Capture failed')

# Save image
image_filename = 'screenshot'
image_suffix = ".png"
script_dir = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(script_dir, "screenshots", image_filename + image_suffix)
print(image_path)

cap.save(image_path)
