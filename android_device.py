# android_device.py - Complete rewrite with connection resilience and input validation
import subprocess as sp
from ppadb.client import Client as AdbClient
from ppadb.device import Device
from PIL import Image
import io
import time
import logging
from typing import Optional, Tuple, List
from contextlib import contextmanager
import numpy as np
from pathlib import Path  # Added missing import

logger = logging.getLogger(__name__)

class DeviceConnection:
    """Manages ADB connection with automatic recovery"""
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self._initialize_adb()
    
    def _initialize_adb(self):
        """Initialize ADB with fixed path"""
        try:
            # Check if adb.exe exists in platform-tools directory
            adb_path = Path(__file__).resolve().parent / 'platform-tools' / 'adb.exe'
            if not adb_path.exists():
                # Try just 'adb' command if platform-tools not found
                adb_command = 'adb'
            else:
                adb_command = str(adb_path)
            
            sp.run([adb_command, 'kill-server'], stdout=sp.PIPE, stderr=sp.PIPE)
            time.sleep(1)
            sp.run([adb_command, 'start-server'], stdout=sp.PIPE, stderr=sp.PIPE)
            time.sleep(2)
            self.client = AdbClient(host="127.0.0.1", port=5037)
        except Exception as e:
            logger.error(f"ADB initialization failed: {e}")
            raise
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device with retry logic"""
        for attempt in range(self.max_retries):
            try:
                devices = self.client.devices()
                
                # Auto-detect if no specific ID
                if device_id == "auto" and devices:
                    return devices[0]
                
                # Find specific device
                for device in devices:
                    if device.serial == device_id:
                        return device
                
                # Device not found, try to connect
                if device_id.startswith("emulator") or ":" in device_id:
                    sp.run(['adb', 'connect', device_id], stdout=sp.PIPE)
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.warning(f"Device connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    self._initialize_adb()
                    
        return None

class AndroidDevice:
    """Enhanced Android device controller with error handling"""
    def __init__(self, device_id: str = "auto"):
        self.device_id = device_id
        self.connection = DeviceConnection()
        self.device = None
        self.screen_size = (720, 1280)  # Default
        self.last_capture_time = 0
        self.min_capture_interval = 0.1
        self._connect()
        
    def _connect(self):
        """Establish device connection"""
        self.device = self.connection.get_device(self.device_id)
        if not self.device:
            raise ConnectionError(f"Failed to connect to device: {self.device_id}")
            
        logger.info(f"Connected to device: {self.device.serial}")
        self._update_screen_size()
        
    def _update_screen_size(self):
        """Get actual screen dimensions"""
        try:
            size_str = self.device.shell("wm size")
            if "Physical size:" in size_str:
                size_part = size_str.split("Physical size:")[1].strip()
                width, height = map(int, size_part.split('x'))
                self.screen_size = (width, height)
                logger.info(f"Screen size: {self.screen_size}")
        except Exception as e:
            logger.warning(f"Failed to get screen size: {e}")
    
    @contextmanager
    def _error_recovery(self):
        """Context manager for automatic error recovery"""
        try:
            yield
        except Exception as e:
            logger.error(f"Device operation failed: {e}")
            logger.info("Attempting to reconnect...")
            self._connect()
            raise
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture screenshot with rate limiting and error handling"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_capture_time < self.min_capture_interval:
            time.sleep(self.min_capture_interval - (current_time - self.last_capture_time))
        
        with self._error_recovery():
            try:
                screen_bytes = self.device.screencap()
                if not screen_bytes:
                    return None
                    
                im = Image.open(io.BytesIO(screen_bytes))
                self.last_capture_time = time.time()
                
                # Convert to numpy array for OpenCV compatibility
                return np.array(im)
                
            except Exception as e:
                logger.error(f"Screenshot capture failed: {e}")
                return None
    
    def tap_xy(self, x: int, y: int, duration: int = 50):
        """Tap with coordinate validation"""
        x = max(0, min(x, self.screen_size[0] - 1))
        y = max(0, min(y, self.screen_size[1] - 1))
        
        with self._error_recovery():
            self.device.input_tap(x, y)
            logger.debug(f"Tapped ({x}, {y})")
    
    def tap_point(self, point: Tuple[int, int], duration: int = 50):
        """Tap point with validation"""
        if point and len(point) >= 2:
            self.tap_xy(point[0], point[1], duration)
    
    def swipe_xy(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """Swipe with coordinate validation"""
        x1 = max(0, min(x1, self.screen_size[0] - 1))
        y1 = max(0, min(y1, self.screen_size[1] - 1))
        x2 = max(0, min(x2, self.screen_size[0] - 1))
        y2 = max(0, min(y2, self.screen_size[1] - 1))
        
        with self._error_recovery():
            self.device.input_swipe(x1, y1, x2, y2, duration)
            logger.debug(f"Swiped ({x1}, {y1}) -> ({x2}, {y2})")
    
    def swipe_point(self, p1: Tuple[int, int], p2: Tuple[int, int], duration: int = 300):
        """Swipe between points"""
        if p1 and p2 and len(p1) >= 2 and len(p2) >= 2:
            self.swipe_xy(p1[0], p1[1], p2[0], p2[1], duration)
    
    def back(self):
        """Press back button"""
        with self._error_recovery():
            self.device.input_keyevent(4)
    
    def home(self):
        """Press home button"""
        with self._error_recovery():
            self.device.input_keyevent(3)
    
    def get_top_activity(self) -> Optional[str]:
        """Get current app package"""
        with self._error_recovery():
            try:
                # Get the dumpsys output for the current activity
                output = self.device.shell("dumpsys activity activities | grep -E 'mResumedActivity|mCurrentFocus'")
                
                # Extract package name from output
                if "com.TechTreeGames.TheTower" in output:
                    return "com.TechTreeGames.TheTower"
                
                # Fallback to checking running packages
                packages = self.device.shell("pm list packages -3")
                if "com.TechTreeGames.TheTower" in packages:
                    # Check if it's actually in foreground
                    return "com.TechTreeGames.TheTower"
                    
                return None
            except Exception as e:
                logger.error(f"Failed to get top activity: {e}")
                return None
    
    def is_app_running(self, package_name: str) -> bool:
        """Check if target app is running"""
        current = self.get_top_activity()
        return current == package_name if current else False
    
    def launch_app(self, package_name: str):
        """Launch app by package name"""
        with self._error_recovery():
            try:
                self.device.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
                logger.info(f"Launched app: {package_name}")
            except Exception as e:
                logger.error(f"Failed to launch app: {e}")
    
    def force_stop_app(self, package_name: str):
        """Force stop application"""
        with self._error_recovery():
            try:
                self.device.shell(f"am force-stop {package_name}")
                logger.info(f"Force stopped app: {package_name}")
            except Exception as e:
                logger.error(f"Failed to stop app: {e}")
