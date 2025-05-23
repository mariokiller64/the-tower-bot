# run_automation.py - Complete rewrite with adaptive timing and recovery
import logging
import signal
import sys
import time
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import deque
import threading
import queue

from android_device import AndroidDevice
from state import create_state_manager, StateManager
from image_operation import extract_game_values, find_button

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler('automation.log'),
       logging.StreamHandler(sys.stdout)
   ]
)
logger = logging.getLogger(__name__)

class AutomationConfig:
   """Configuration management with hot-reload support"""
   def __init__(self, config_file: str = "config.json"):
       self.config_file = Path(config_file)
       self.load_defaults()
       self.load_from_file()
       self._last_modified = self.config_file.stat().st_mtime if self.config_file.exists() else 0
   
   def load_defaults(self):
       """Set default configuration"""
       self.app_package = 'com.TechTreeGames.TheTower'
       self.device_id = 'auto'
       self.base_delay = 1.0
       self.adaptive_timing = True
       self.max_unknown_states = 10
       self.screenshot_dir = Path("screenshots")
       self.save_unknown_states = True
       self.performance_mode = "balanced"  # "fast", "balanced", "quality"
       self.enable_ocr = False
       self.enable_predictions = True
       self.multi_threading = True
       self.state_config_path = Path("state_config.json")
       
   def load_from_file(self):
       """Load configuration from JSON file"""
       try:
           if self.config_file.exists():
               with open(self.config_file, 'r') as f:
                   data = json.load(f)
                   for key, value in data.items():
                       if hasattr(self, key):
                           # Convert Path strings back to Path objects
                           if key in ['screenshot_dir', 'state_config_path'] and isinstance(value, str):
                               value = Path(value)
                           setattr(self, key, value)
               logger.info(f"Loaded config from {self.config_file}")
       except Exception as e:
           logger.warning(f"Failed to load config: {e}")
   
   def check_reload(self):
       """Check if config file has been modified and reload if needed"""
       if not self.config_file.exists():
           return False
           
       current_mtime = self.config_file.stat().st_mtime
       if current_mtime > self._last_modified:
           logger.info("Config file changed, reloading...")
           self.load_from_file()
           self._last_modified = current_mtime
           return True
       return False
   
   def save_to_file(self):
       """Save current configuration"""
       data = {}
       for key, value in self.__dict__.items():
           if key.startswith('_'):
               continue
           # Convert Path objects to strings for JSON serialization
           if isinstance(value, Path):
               value = str(value)
           data[key] = value
           
       with open(self.config_file, 'w') as f:
           json.dump(data, f, indent=2)
       self._last_modified = self.config_file.stat().st_mtime

class PerformanceMonitor:
   """Monitor and optimize automation performance"""
   def __init__(self, window_size: int = 100):
       self.window_size = window_size
       self.cycle_times = deque(maxlen=window_size)
       self.capture_times = deque(maxlen=window_size)
       self.process_times = deque(maxlen=window_size)
       self.state_confidences = deque(maxlen=window_size)
       self.memory_usage = deque(maxlen=window_size)
       
   def record_cycle(self, cycle_time: float, capture_time: float, 
                   process_time: float, confidence: float = 1.0):
       """Record performance metrics for a cycle"""
       self.cycle_times.append(cycle_time)
       self.capture_times.append(capture_time)
       self.process_times.append(process_time)
       self.state_confidences.append(confidence)
       
       # Track memory usage
       try:
           import psutil
           process = psutil.Process()
           memory_mb = process.memory_info().rss / 1024 / 1024
           self.memory_usage.append(memory_mb)
       except ImportError:
           pass
   
   def get_stats(self) -> Dict[str, float]:
       """Get performance statistics"""
       stats = {}
       
       if self.cycle_times:
           stats['avg_cycle_time'] = np.mean(self.cycle_times)
           stats['min_cycle_time'] = np.min(self.cycle_times)
           stats['max_cycle_time'] = np.max(self.cycle_times)
           
       if self.capture_times:
           stats['avg_capture_time'] = np.mean(self.capture_times)
           
       if self.process_times:
           stats['avg_process_time'] = np.mean(self.process_times)
           
       if self.state_confidences:
           stats['avg_confidence'] = np.mean(self.state_confidences)
           stats['min_confidence'] = np.min(self.state_confidences)
           
       if self.memory_usage:
           stats['avg_memory_mb'] = np.mean(self.memory_usage)
           stats['max_memory_mb'] = np.max(self.memory_usage)
           
       return stats
   
   def suggest_optimizations(self) -> List[str]:
       """Suggest performance optimizations based on metrics"""
       suggestions = []
       stats = self.get_stats()
       
       if stats.get('avg_cycle_time', 0) > 2.0:
           suggestions.append("Consider reducing image processing quality for faster cycles")
           
       if stats.get('avg_capture_time', 0) > 0.5:
           suggestions.append("Screenshot capture is slow - check ADB connection")
           
       if stats.get('avg_confidence', 1.0) < 0.7:
           suggestions.append("Low detection confidence - update reference images")
           
       if stats.get('max_memory_mb', 0) > 500:
           suggestions.append("High memory usage - consider restarting periodically")
           
       return suggestions

class AdaptiveTiming:
   """Intelligent delay management based on game state and performance"""
   def __init__(self, base_delay: float = 1.0):
       self.base_delay = base_delay
       self.state_delays = {
           'MenuState': 2.0,
           'GameOverState': 3.0,
           'PlayAttackState': 0.5,
           'PlayDefenseState': 0.5,
           'PlayUtilityState': 0.5,
       }
       self.performance_factor = 1.0
       self.confidence_factor = 1.0
       self.recent_errors = deque(maxlen=10)
       
   def get_delay(self, state_name: Optional[str], confidence: float = 1.0) -> float:
       """Calculate adaptive delay for current state"""
       # Base delay for state
       if not state_name:
           base = self.base_delay * 2  # Longer delay when lost
       else:
           base = self.state_delays.get(state_name, self.base_delay)
       
       # Adjust based on confidence
       if confidence < 0.9:
           self.confidence_factor = min(1.5, self.confidence_factor + 0.1)
       else:
           self.confidence_factor = max(0.8, self.confidence_factor - 0.05)
       
       # Adjust based on recent errors
       error_rate = sum(self.recent_errors) / max(1, len(self.recent_errors))
       if error_rate > 0.3:
           self.performance_factor = min(2.0, self.performance_factor + 0.1)
       else:
           self.performance_factor = max(0.5, self.performance_factor - 0.05)
       
       return base * self.performance_factor * self.confidence_factor
   
   def record_error(self, had_error: bool):
       """Track error rate for adjustment"""
       self.recent_errors.append(1 if had_error else 0)

class AsyncScreenCapture:
   """Asynchronous screenshot capture for better performance"""
   def __init__(self, device: AndroidDevice):
       self.device = device
       self.capture_queue = queue.Queue(maxsize=2)
       self.running = False
       self.thread = None
       
   def start(self):
       """Start async capture thread"""
       self.running = True
       self.thread = threading.Thread(target=self._capture_loop, daemon=True)
       self.thread.start()
       
   def stop(self):
       """Stop capture thread"""
       self.running = False
       if self.thread:
           self.thread.join(timeout=2)
           
   def _capture_loop(self):
       """Continuous capture loop"""
       while self.running:
           try:
               capture = self.device.capture()
               if capture is not None:
                   # Discard old capture if queue is full
                   if self.capture_queue.full():
                       try:
                           self.capture_queue.get_nowait()
                       except queue.Empty:
                           pass
                   
                   self.capture_queue.put((time.time(), capture))
               else:
                   time.sleep(0.1)
                   
           except Exception as e:
               logger.error(f"Capture thread error: {e}")
               time.sleep(1)
   
   def get_latest(self, timeout: float = 1.0) -> Optional[Tuple[float, np.ndarray]]:
       """Get latest capture with timestamp"""
       try:
           return self.capture_queue.get(timeout=timeout)
       except queue.Empty:
           return None

class TowerAutomation:
   """Main automation controller with advanced features"""
   def __init__(self, config: AutomationConfig):
       self.config = config
       self.device = None
       self.state_manager = create_state_manager(
           config.state_config_path if config.state_config_path.exists() else None
       )
       self.timing = AdaptiveTiming(config.base_delay)
       self.performance = PerformanceMonitor()
       self.async_capture = None
       self.running = False
       
       # Statistics
       self.stats = {
           'start_time': time.time(),
           'cycles': 0,
           'errors': 0,
           'unknown_states': 0,
           'captures_failed': 0,
           'actions_performed': 0,
       }
       
       # Game value tracking
       self.game_values_history = deque(maxlen=100)
       
       # Setup signal handlers
       signal.signal(signal.SIGINT, self._signal_handler)
       signal.signal(signal.SIGTERM, self._signal_handler)
       
       # Create directories
       self.config.screenshot_dir.mkdir(exist_ok=True)
       (self.config.screenshot_dir / "unknown_states").mkdir(exist_ok=True)
       
   def _signal_handler(self, sig, frame):
       """Handle shutdown signals gracefully"""
       logger.info("Shutdown signal received")
       self.stop()
       
   def connect_device(self) -> bool:
       """Establish device connection"""
       try:
           self.device = AndroidDevice(self.config.device_id)
           
           # Start async capture if enabled
           if self.config.multi_threading:
               self.async_capture = AsyncScreenCapture(self.device)
               self.async_capture.start()
               logger.info("Started asynchronous capture")
               
           return True
       except Exception as e:
           logger.error(f"Failed to connect to device: {e}")
           return False
   
   def ensure_app_running(self) -> bool:
       """Ensure target app is in foreground"""
       if not self.device.is_app_running(self.config.app_package):
           logger.info("App not running, launching...")
           self.device.launch_app(self.config.app_package)
           time.sleep(5)  # Wait for app to start
           
           # Verify launch
           if not self.device.is_app_running(self.config.app_package):
               logger.error("Failed to launch app")
               return False
               
       return True
   
   def save_unknown_state(self, capture: np.ndarray, confidence_scores: Dict[str, float]):
       """Save screenshot of unknown state for analysis"""
       if not self.config.save_unknown_states:
           return
           
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
       
       # Save screenshot
       screenshot_path = self.config.screenshot_dir / "unknown_states" / f"unknown_{timestamp}.png"
       try:
           import cv2
           cv2.imwrite(str(screenshot_path), cv2.cvtColor(capture, cv2.COLOR_RGB2BGR))
           
           # Save metadata
           metadata = {
               'timestamp': timestamp,
               'confidence_scores': confidence_scores,
               'game_values': self.game_values_history[-1] if self.game_values_history else {},
               'previous_state': self.state_manager.game_state.last_state.name 
                   if self.state_manager.game_state.last_state else None,
           }
           
           metadata_path = screenshot_path.with_suffix('.json')
           with open(metadata_path, 'w') as f:
               json.dump(metadata, f, indent=2)
               
           logger.info(f"Saved unknown state to {screenshot_path}")
           
       except Exception as e:
           logger.error(f"Failed to save unknown state: {e}")
   
   def extract_game_values_safe(self, capture: np.ndarray) -> Dict[str, any]:
       """Safely extract game values with fallback"""
       if not self.config.enable_ocr:
           return {}
           
       try:
           values = extract_game_values(capture)
           self.game_values_history.append(values)
           
           # Update game state with extracted values
           if 'health' in values:
               self.state_manager.game_state.health = values['health']
           if 'gold' in values:
               self.state_manager.game_state.gold = values['gold']
           if 'wave' in values:
               self.state_manager.game_state.wave_number = values['wave']
               
           return values
           
       except Exception as e:
           logger.error(f"Failed to extract game values: {e}")
           return {}
   
   def run_cycle(self):
       """Execute one automation cycle with performance tracking"""
       cycle_start = time.time()
       had_error = False
       
       try:
           # Get screenshot
           capture_start = time.time()
           
           if self.async_capture:
               # Use async capture
               result = self.async_capture.get_latest()
               if result:
                   capture_timestamp, capture = result
                   # Check if capture is too old
                   if time.time() - capture_timestamp > 2.0:
                       capture = self.device.capture()
               else:
                   capture = self.device.capture()
           else:
               # Synchronous capture
               capture = self.device.capture()
               
           capture_time = time.time() - capture_start
           
           if capture is None:
               logger.error("Capture failed")
               self.stats['captures_failed'] += 1
               self.stats['errors'] += 1
               had_error = True
               return
           
           # Process state and execute actions
           process_start = time.time()
           
           # Extract game values if enabled
           if self.config.enable_ocr:
               self.extract_game_values_safe(capture)
           
           # Detect current state
           current_state = self.state_manager.detect_state(capture)
           
           # Get confidence scores for logging
           confidence = 0.0
           confidence_scores = {}
           if hasattr(self.state_manager, 'confidence_history') and self.state_manager.confidence_history:
               latest = self.state_manager.confidence_history[-1]
               confidence = latest.get('confidence', 0.0)
               confidence_scores = latest.get('all_confidences', {})
           
           if current_state:
               logger.info(f"State: {current_state.name} (confidence: {confidence:.2f})")
               self.state_manager.execute_current_state(self.device, capture)
               self.stats['actions_performed'] += 1
               self.stats['unknown_states'] = 0
           else:
               self.stats['unknown_states'] += 1
               logger.warning(f"Unknown state ({self.stats['unknown_states']} consecutive)")
               
               if self.stats['unknown_states'] >= self.config.max_unknown_states:
                   self.save_unknown_state(capture, confidence_scores)
                   self.handle_lost_state()
           
           process_time = time.time() - process_start
           
           # Record performance metrics
           cycle_time = time.time() - cycle_start
           self.performance.record_cycle(cycle_time, capture_time, process_time, confidence)
           
           # Adaptive delay
           delay = self.timing.get_delay(
               current_state.name if current_state else None,
               confidence
           )
           
           if cycle_time < delay:
               time.sleep(delay - cycle_time)
           
           self.stats['cycles'] += 1
           
       except Exception as e:
           logger.error(f"Cycle error: {e}", exc_info=True)
           self.stats['errors'] += 1
           had_error = True
           
       finally:
           self.timing.record_error(had_error)
   
   def handle_lost_state(self):
       """Advanced recovery when bot is lost"""
       logger.warning("Bot is lost, attempting recovery")
       
       # Recovery strategies based on performance mode
       if self.config.performance_mode == "fast":
           # Quick recovery - just tap center and continue
           self.device.tap_xy(360, 640)
           time.sleep(1)
       else:
           # Comprehensive recovery
           recovery_strategies = [
               # Strategy 1: Try common UI locations
               lambda: self._try_common_buttons(),
               
               # Strategy 2: Check if we're in a dialog
               lambda: self._check_dialogs(),
               
               # Strategy 3: Try going back
               lambda: self.device.back(),
               lambda: time.sleep(2),
               
               # Strategy 4: Verify app is running
               lambda: self.ensure_app_running(),
               
               # Strategy 5: Force restart if nothing works
               lambda: self._force_restart_app() if self.stats['unknown_states'] > 20 else None,
           ]
           
           for strategy in recovery_strategies:
               try:
                   result = strategy()
                   if result:  # Some strategies return True if successful
                       break
               except Exception as e:
                   logger.error(f"Recovery strategy failed: {e}")
       
       self.stats['unknown_states'] = 0
   
   def _try_common_buttons(self) -> bool:
       """Try to find and click common buttons"""
       capture = self.device.capture()
       if capture is None:
           return False
           
       # Try to find common buttons
       for button_name in ['close', 'ok', 'continue', 'retry', 'back']:
           button_pos = find_button(capture, button_name)
           if button_pos:
               logger.info(f"Found {button_name} button")
               self.device.tap_point(button_pos)
               time.sleep(1)
               return True
               
       return False
   
   def _check_dialogs(self) -> bool:
       """Check for and handle common dialogs"""
       # Common dialog close button locations
       dialog_close_positions = [
           (650, 200),  # Top right
           (360, 1000), # Bottom center
           (600, 400),  # Middle right
       ]
       
       for pos in dialog_close_positions:
           self.device.tap_point(pos)
           time.sleep(0.5)
           
       return False
   
   def _force_restart_app(self):
       """Force restart the app as last resort"""
       logger.warning("Force restarting app")
       self.device.force_stop_app(self.config.app_package)
       time.sleep(2)
       self.device.launch_app(self.config.app_package)
       time.sleep(5)
   
   def print_stats(self):
       """Display comprehensive automation statistics"""
       runtime = time.time() - self.stats['start_time']
       runtime_hours = runtime / 3600
       
       logger.info("\n" + "="*50)
       logger.info("AUTOMATION STATISTICS")
       logger.info("="*50)
       
       # Basic stats
       logger.info(f"Runtime: {runtime_hours:.2f} hours ({runtime:.0f} seconds)")
       logger.info(f"Total cycles: {self.stats['cycles']}")
       logger.info(f"Actions performed: {self.stats['actions_performed']}")
       logger.info(f"Errors: {self.stats['errors']}")
       logger.info(f"Capture failures: {self.stats['captures_failed']}")
       
       # Performance stats
       perf_stats = self.performance.get_stats()
       if perf_stats:
           logger.info("\nPERFORMANCE METRICS:")
           logger.info(f"Avg cycle time: {perf_stats.get('avg_cycle_time', 0):.2f}s")
           logger.info(f"Avg capture time: {perf_stats.get('avg_capture_time', 0):.3f}s")
           logger.info(f"Avg process time: {perf_stats.get('avg_process_time', 0):.3f}s")
           logger.info(f"Avg confidence: {perf_stats.get('avg_confidence', 0):.2%}")
           
           if 'avg_memory_mb' in perf_stats:
               logger.info(f"Avg memory usage: {perf_stats['avg_memory_mb']:.1f} MB")
       
       # Game stats
       game_state = self.state_manager.game_state
       logger.info("\nGAME STATISTICS:")
       logger.info(f"Total runs: {game_state.total_runs}")
       logger.info(f"Best wave: {game_state.best_wave}")
       logger.info(f"Current wave: {game_state.wave_number}")
       
       if game_state.upgrade_counts:
           logger.info("\nUPGRADE DISTRIBUTION:")
           total_upgrades = sum(game_state.upgrade_counts.values())
           for upgrade, count in sorted(game_state.upgrade_counts.items(), 
                                      key=lambda x: x[1], reverse=True):
               percentage = (count / total_upgrades) * 100
               logger.info(f"  {upgrade}: {count} ({percentage:.1f}%)")
       
       # Performance suggestions
       suggestions = self.performance.suggest_optimizations()
       if suggestions:
           logger.info("\nPERFORMANCE SUGGESTIONS:")
           for suggestion in suggestions:
               logger.info(f"  • {suggestion}")
       
       logger.info("="*50 + "\n")
   
   def run(self):
       """Main automation loop with hot-reload and monitoring"""
       logger.info("Starting Tower Defense Bot Automation")
       logger.info(f"Performance mode: {self.config.performance_mode}")
       logger.info(f"Device: {self.config.device_id}")
       
       if not self.connect_device():
           return
       
       if not self.ensure_app_running():
           logger.error("Failed to start app")
           return
       
       self.running = True
       last_stats_print = time.time()
       last_config_check = time.time()
       consecutive_errors = 0
       
       try:
           while self.running:
               try:
                   # Run automation cycle
                   self.run_cycle()
                   consecutive_errors = 0
                   
                   # Periodic tasks
                   current_time = time.time()
                   
                   # Print stats every 5 minutes
                   if current_time - last_stats_print > 300:
                       self.print_stats()
                       last_stats_print = current_time
                   
                   # Check for config changes every 30 seconds
                   if current_time - last_config_check > 30:
                       if self.config.check_reload():
                           # Apply new settings
                           self.timing.base_delay = self.config.base_delay
                           logger.info("Applied new configuration")
                       last_config_check = current_time
                   
               except Exception as e:
                   logger.error(f"Cycle error: {e}")
                   self.stats['errors'] += 1
                   consecutive_errors += 1
                   
                   # Exponential backoff for consecutive errors
                   wait_time = min(30, 2 ** consecutive_errors)
                   logger.info(f"Waiting {wait_time}s before retry...")
                   time.sleep(wait_time)
                   
                   # Reconnect if too many consecutive errors
                   if consecutive_errors > 5:
                       logger.warning("Too many consecutive errors, reconnecting")
                       self.connect_device()
                       self.ensure_app_running()
                       consecutive_errors = 0
                       
       except KeyboardInterrupt:
           logger.info("Automation interrupted by user")
       finally:
           self.cleanup()
   
   def cleanup(self):
       """Clean up resources and save state"""
       logger.info("Cleaning up...")
       
       # Stop async capture
       if self.async_capture:
           self.async_capture.stop()
       
       # Save statistics
       self.print_stats()
       
       # Save state manager config
       if self.config.state_config_path:
           self.state_manager.save_config(self.config.state_config_path)
           logger.info(f"Saved state configuration to {self.config.state_config_path}")
       
       # Save automation config
       self.config.save_to_file()
       logger.info(f"Saved automation config to {self.config.config_file}")
       
       # Save final statistics
       stats_path = self.config.screenshot_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
       with open(stats_path, 'w') as f:
           json.dump({
               'stats': self.stats,
               'performance': self.performance.get_stats(),
               'game_state': {
                   'total_runs': self.state_manager.game_state.total_runs,
                   'best_wave': self.state_manager.game_state.best_wave,
                   'upgrade_counts': self.state_manager.game_state.upgrade_counts,
               }
           }, f, indent=2)
       logger.info(f"Saved statistics to {stats_path}")
   
   def stop(self):
       """Stop automation gracefully"""
       self.running = False
       logger.info("Stopping automation...")

def main():
   """Entry point with argument parsing"""
   import argparse
   
   parser = argparse.ArgumentParser(description="Tower Defense Bot Automation")
   parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
   parser.add_argument('--device', type=str, help='Override device ID')
   parser.add_argument('--performance', choices=['fast', 'balanced', 'quality'],
                      help='Override performance mode')
   parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
   
   args = parser.parse_args()
   
   # Set debug logging if requested
   if args.debug:
       logging.getLogger().setLevel(logging.DEBUG)
   
   # Load configuration
   config = AutomationConfig(args.config)
   
   # Apply command line overrides
   if args.device:
       config.device_id = args.device
   if args.performance:
       config.performance_mode = args.performance
   
   # Create and run automation
   automation = TowerAutomation(config)
   
   try:
       automation.run()
   except Exception as e:
       logger.error(f"Fatal error: {e}", exc_info=True)
       sys.exit(1)

if __name__ == "__main__":
   main()
