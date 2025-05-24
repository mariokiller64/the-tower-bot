# run_automation.py - Gem farming focused automation
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
from state import create_state_manager, StateManager, AdGemState
from image_operation import extract_game_values, find_button, detect_spinning_gem

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
   """Configuration for gem-focused automation"""
   def __init__(self, config_file: str = "config.json"):
       self.config_file = Path(config_file)
       self.load_defaults()
       self.load_from_file()
       self._last_modified = self.config_file.stat().st_mtime if self.config_file.exists() else 0
   
   def load_defaults(self):
       """Set default configuration for gem farming"""
       self.app_package = 'com.techtreegames.thetower'
       self.device_id = 'auto'
       
       # Timing settings for gem farming
       self.main_loop_interval = 60  # Check every 60 seconds (user configurable)
       self.ad_check_interval = 600  # Check for ad every 10 minutes
       self.spinning_gem_check_interval = 15  # Check for spinning gem every 15 seconds in wave
       self.ad_watch_duration = 60  # 60 second ads
       
       # Features
       self.enable_ocr = True
       self.gem_farming_mode = True
       self.auto_upgrades = True
       self.auto_wave_start = True
       
       # Paths
       self.screenshot_dir = Path("screenshots")
       self.template_dir = Path("templates")
       self.state_config_path = Path("state_config.json")
       
       # Performance
       self.save_debug_screenshots = False
       self.randomize_taps = True
       self.tap_delay_range = (0.5, 1.0)
       
   def load_from_file(self):
       """Load configuration from JSON file"""
       try:
           if self.config_file.exists():
               with open(self.config_file, 'r') as f:
                   data = json.load(f)
                   for key, value in data.items():
                       if hasattr(self, key):
                           # Convert Path strings back to Path objects
                           if key.endswith('_dir') or key.endswith('_path'):
                               value = Path(value)
                           setattr(self, key, value)
               logger.info(f"Loaded config from {self.config_file}")
       except Exception as e:
           logger.warning(f"Failed to load config: {e}")
   
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

class GemFarmingStats:
   """Track gem farming performance"""
   def __init__(self):
       self.session_start = time.time()
       self.gems_start = 0
       self.gems_current = 0
       self.ads_watched = 0
       self.spinning_gems_collected = 0
       self.ad_attempts = 0
       self.ad_failures = 0
       self.last_ad_time = 0
       self.gem_history = deque(maxlen=100)
       
   def update_gems(self, new_gem_count: int):
       """Update gem count and track changes"""
       if new_gem_count > self.gems_current:
           gain = new_gem_count - self.gems_current
           self.gem_history.append({
               'time': time.time(),
               'amount': gain,
               'total': new_gem_count
           })
       self.gems_current = new_gem_count
   
   def get_gems_per_hour(self) -> float:
       """Calculate gems per hour rate"""
       runtime = time.time() - self.session_start
       if runtime < 60:  # Need at least 1 minute
           return 0.0
       
       total_gained = self.gems_current - self.gems_start
       hours = runtime / 3600
       return total_gained / hours if hours > 0 else 0.0
   
   def get_summary(self) -> Dict[str, any]:
       """Get farming statistics summary"""
       runtime = time.time() - self.session_start
       total_gained = self.gems_current - self.gems_start
       
       return {
           'runtime_hours': runtime / 3600,
           'total_gems_gained': total_gained,
           'gems_per_hour': self.get_gems_per_hour(),
           'ads_watched': self.ads_watched,
           'ad_success_rate': (self.ads_watched / max(1, self.ad_attempts)) * 100,
           'spinning_gems': self.spinning_gems_collected,
           'current_gems': self.gems_current,
       }

class TowerAutomation:
   """Main automation controller focused on gem farming"""
   def __init__(self, config: AutomationConfig):
       self.config = config
       self.device = None
       self.state_manager = create_state_manager(
           config.state_config_path if config.state_config_path.exists() else None
       )
       self.running = False
       self.gem_stats = GemFarmingStats()
       
       # Timing trackers
       self.last_main_loop = 0
       self.last_ad_check = 0
       self.last_spinning_check = 0
       self.last_ad_success = 0
       self.ad_retry_count = 0
       
       # State tracking
       self.in_wave = False
       self.watching_ad = False
       
       # Setup signal handlers
       signal.signal(signal.SIGINT, self._signal_handler)
       signal.signal(signal.SIGTERM, self._signal_handler)
       
       # Create directories
       self.config.screenshot_dir.mkdir(exist_ok=True)
       self.config.template_dir.mkdir(exist_ok=True)
       
   def _signal_handler(self, sig, frame):
       """Handle shutdown signals gracefully"""
       logger.info("Shutdown signal received")
       self.stop()
       
   def connect_device(self) -> bool:
       """Establish device connection"""
       try:
           self.device = AndroidDevice(self.config.device_id)
           logger.info(f"Connected to device: {self.config.device_id}")
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
   
   def check_ad_gem(self) -> bool:
       """Priority 1: Check and claim 5-gem ad"""
       logger.info("🔍 Checking for ad gem button...")
       self.gem_stats.ad_attempts += 1
       
       try:
           # Take screenshot
           capture = self.device.capture()
           if capture is None:
               logger.error("Failed to capture screen")
               return False
           
           # Extract current gems for verification
           values = extract_game_values(capture)
           gems_before = values.get('gems', self.gem_stats.gems_current)
           self.gem_stats.update_gems(gems_before)
           
           # Look for ad button
           ad_button = find_button(capture, "5_gem_ad")
           
           if not ad_button:
               logger.info("No ad button found")
               return False
           
           logger.info(f"📺 Found 5-gem ad button at {ad_button}!")
           
           # Tap ad button with randomization
           if self.config.randomize_taps:
               tap_x = ad_button[0] + np.random.randint(-10, 10)
               tap_y = ad_button[1] + np.random.randint(-10, 10)
           else:
               tap_x, tap_y = ad_button
           
           self.device.tap_xy(tap_x, tap_y)
           time.sleep(2)  # Wait for ad to start
           
           # Watch ad
           self.watching_ad = True
           logger.info(f"⏳ Watching ad for {self.config.ad_watch_duration} seconds...")
           
           # Show countdown
           for remaining in range(self.config.ad_watch_duration, 0, -10):
               logger.info(f"   {remaining} seconds remaining...")
               time.sleep(min(10, remaining))
           
           self.watching_ad = False
           
           # Try multiple methods to close ad
           logger.info("✅ Ad finished, claiming reward...")
           time.sleep(1)
           
           # Take new screenshot for close button
           capture = self.device.capture()
           if capture:
               # Try to find specific close buttons
               for button_name in ['claim', 'close', 'x_button']:
                   close_button = find_button(capture, button_name)
                   if close_button:
                       self.device.tap_point(close_button)
                       time.sleep(0.5)
           
           # Fallback: tap common close positions
           close_positions = [
               (360, 800),   # Center claim
               (650, 100),   # Top-right X
               (360, 1100),  # Bottom close
               (50, 50),     # Top-left back
           ]
           
           for pos in close_positions:
               self.device.tap_point(pos)
               time.sleep(0.3)
           
           # Verify gem increase
           time.sleep(2)
           capture = self.device.capture()
           if capture:
               values = extract_game_values(capture)
               gems_after = values.get('gems', gems_before)
               self.gem_stats.update_gems(gems_after)
               
               if gems_after > gems_before:
                   gem_gain = gems_after - gems_before
                   self.gem_stats.ads_watched += 1
                   self.last_ad_success = time.time()
                   self.ad_retry_count = 0
                   logger.info(f"💎 SUCCESS! Gained {gem_gain} gems! Total: {gems_after}")
                   
                   # Update game state
                   self.state_manager.game_state.track_gem_collection(gem_gain, "ad")
                   
                   return True
               else:
                   logger.warning("⚠️ Could not verify gem increase")
                   self.gem_stats.ad_failures += 1
           
       except Exception as e:
           logger.error(f"Ad gem check failed: {e}")
           self.gem_stats.ad_failures += 1
       
       return False
   
   def check_spinning_gem(self) -> bool:
       """Priority 2: Check for spinning gem during waves"""
       if not self.in_wave:
           return False
       
       try:
           # Take screenshot
           capture = self.device.capture()
           if capture is None:
               return False
           
           # Detect spinning gem
           gem_position = detect_spinning_gem(capture)
           
           if gem_position:
               logger.info(f"🔹 Found spinning gem at {gem_position}!")
               
               # Tap with randomization
               if self.config.randomize_taps:
                   tap_x = gem_position[0] + np.random.randint(-5, 5)
                   tap_y = gem_position[1] + np.random.randint(-5, 5)
               else:
                   tap_x, tap_y = gem_position
               
               self.device.tap_xy(tap_x, tap_y)
               
               # Track collection
               self.gem_stats.spinning_gems_collected += 1
               self.gem_stats.update_gems(self.gem_stats.gems_current + 2)
               self.state_manager.game_state.track_gem_collection(2, "spinning")
               
               return True
               
       except Exception as e:
           logger.error(f"Spinning gem check failed: {e}")
       
       return False
   
   def run_main_cycle(self):
       """Execute main automation cycle"""
       try:
           # Take screenshot
           capture = self.device.capture()
           if capture is None:
               logger.error("Capture failed")
               return
           
           # Extract and update game values
           if self.config.enable_ocr:
               values = extract_game_values(capture)
               if values:
                   # Update gem count
                   if 'gems' in values:
                       self.gem_stats.update_gems(values['gems'])
                   
                   # Update game state
                   game_state = self.state_manager.game_state
                   game_state.gems = values.get('gems', game_state.gems)
                   game_state.gold = values.get('gold', game_state.gold)
                   game_state.wave_number = values.get('wave', game_state.wave_number)
           
           # Detect and execute current state
           current_state = self.state_manager.detect_state(capture)
           
           if current_state:
               logger.info(f"📍 State: {current_state.name}")
               
               # Update wave status
               self.in_wave = current_state.name in ['PlayAttackState', 'PlayDefenseState', 'PlayUtilityState']
               
               # Execute state strategy
               self.state_manager.execute_current_state(self.device, capture)
           else:
               logger.warning("Unknown state")
               
               # Try to recover
               if self.state_manager.unknown_state_count > 5:
                   logger.warning("Lost for too long, attempting recovery")
                   self.device.tap_xy(360, 640)  # Tap center
                   time.sleep(1)
                   self.device.back()  # Try back button
                   
       except Exception as e:
           logger.error(f"Main cycle error: {e}")
   
   def run(self):
       """Main gem farming loop"""
       logger.info("🎮 Starting Tower Defense Gem Farming Bot")
       logger.info(f"⚙️ Main loop: {self.config.main_loop_interval}s")
       logger.info(f"⚙️ Ad check: {self.config.ad_check_interval}s")
       logger.info(f"⚙️ Spinning gem check: {self.config.spinning_gem_check_interval}s")
       
       if not self.connect_device():
           return
       
       if not self.ensure_app_running():
           logger.error("Failed to start app")
           return
       
       # Get initial gem count
       time.sleep(2)
       capture = self.device.capture()
       if capture:
           values = extract_game_values(capture)
           if 'gems' in values:
               self.gem_stats.gems_start = values['gems']
               self.gem_stats.gems_current = values['gems']
               logger.info(f"💎 Starting gems: {self.gem_stats.gems_start}")
       
       self.running = True
       
       # Start CLI interface in separate thread
       cli_thread = threading.Thread(target=self.run_cli_interface, daemon=True)
       cli_thread.start()
       
       try:
           while self.running:
               current_time = time.time()
               
               # Priority 1: Ad gem check (every 10 minutes)
               if current_time - self.last_ad_check >= self.config.ad_check_interval:
                   if not self.watching_ad:  # Don't check while watching ad
                       self.check_ad_gem()
                       self.last_ad_check = current_time
                   
                   # If ad failed, retry more frequently
                   if self.ad_retry_count > 0 and current_time - self.last_ad_success > 900:
                       # No ad success in 15 minutes, retry every 30 seconds
                       self.config.ad_check_interval = 30
                       self.ad_retry_count += 1
                   else:
                       # Reset to normal interval
                       self.config.ad_check_interval = 600
               
               # Priority 2: Spinning gem check (every 15 seconds if in wave)
               if self.in_wave and current_time - self.last_spinning_check >= self.config.spinning_gem_check_interval:
                   self.check_spinning_gem()
                   self.last_spinning_check = current_time
               
               # Regular automation cycle (every 60 seconds)
               if current_time - self.last_main_loop >= self.config.main_loop_interval:
                   self.run_main_cycle()
                   self.last_main_loop = current_time
               
               # Sleep to prevent CPU waste
               time.sleep(1)
               
       except KeyboardInterrupt:
           logger.info("Automation interrupted by user")
       finally:
           self.cleanup()
   
   def run_cli_interface(self):
       """Simple command-line interface"""
       print("\n" + "="*50)
       print("💎 TOWER DEFENSE GEM FARMING BOT 💎")
       print("="*50)
       print("\nCommands:")
       print("  stats  - Show current statistics")
       print("  freq   - Change check frequency")
       print("  stop   - Stop bot")
       print("  help   - Show commands")
       print("\n")
       
       while self.running:
           try:
               cmd = input("> ").strip().lower()
               
               if cmd == "stats":
                   self.print_stats()
               elif cmd == "freq":
                   self.change_frequency()
               elif cmd == "stop":
                   self.stop()
               elif cmd == "help":
                   print("\nCommands: stats, freq, stop, help\n")
               elif cmd:
                   print("Unknown command. Type 'help' for commands.")
                   
           except EOFError:
               # Handle Ctrl+D
               break
           except Exception:
               # Ignore other errors in CLI
               pass
   
   def print_stats(self):
       """Display current statistics"""
       stats = self.gem_stats.get_summary()
       game_state = self.state_manager.game_state
       
       print("\n" + "="*50)
       print("📊 GEM FARMING STATISTICS")
       print("="*50)
       
       print(f"⏱️ Runtime: {stats['runtime_hours']:.2f} hours")
       print(f"💎 Current Gems: {stats['current_gems']}")
       print(f"📈 Gems Gained: {stats['total_gems_gained']}")
       print(f"⚡ Gems/Hour: {stats['gems_per_hour']:.1f}")
       print(f"📺 Ads Watched: {stats['ads_watched']}")
       print(f"✅ Ad Success Rate: {stats['ad_success_rate']:.1f}%")
       print(f"🔹 Spinning Gems: {stats['spinning_gems']}")
       print(f"🌊 Current Wave: {game_state.wave_number}")
       print(f"🏆 Best Wave: {game_state.best_wave}")
       
       # Next ad time
       next_ad = self.config.ad_check_interval - (time.time() - self.last_ad_check)
       if next_ad > 0:
           print(f"⏰ Next Ad Check: {next_ad/60:.1f} minutes")
       else:
           print(f"⏰ Next Ad Check: Now!")
       
       print("="*50 + "\n")
   
   def change_frequency(self):
       """Change check frequency"""
       print(f"\nCurrent check frequency: {self.config.main_loop_interval} seconds")
       print("Enter new frequency (30-120 seconds):")
       
       try:
           new_freq = int(input("New frequency: "))
           if 30 <= new_freq <= 120:
               self.config.main_loop_interval = new_freq
               print(f"✅ Check frequency set to {new_freq} seconds")
           else:
               print("❌ Frequency must be between 30-120 seconds")
       except ValueError:
           print("❌ Invalid number")
   
   def cleanup(self):
       """Clean up resources and save state"""
       logger.info("Cleaning up...")
       
       # Final statistics
       self.print_stats()
       
       # Save state manager config
       if self.config.state_config_path:
           self.state_manager.save_config(self.config.state_config_path)
           logger.info(f"Saved state configuration")
       
       # Save automation config
       self.config.save_to_file()
       logger.info(f"Saved automation config")
       
       # Save final statistics
       stats_path = self.config.screenshot_dir / f"gem_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
       with open(stats_path, 'w') as f:
           json.dump({
               'stats': self.gem_stats.get_summary(),
               'game_state': {
                   'total_runs': self.state_manager.game_state.total_runs,
                   'best_wave': self.state_manager.game_state.best_wave,
                   'total_gems_collected': self.state_manager.game_state.gems_collected,
               },
               'timestamp': datetime.now().isoformat()
           }, f, indent=2)
       logger.info(f"Saved statistics to {stats_path}")
   
   def stop(self):
       """Stop automation gracefully"""
       self.running = False
       logger.info("Stopping automation...")

def main():
   """Entry point"""
   import argparse
   
   parser = argparse.ArgumentParser(description="Tower Defense Gem Farming Bot")
   parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
   parser.add_argument('--device', type=str, help='Override device ID')
   parser.add_argument('--loop-interval', type=int, help='Main loop interval (30-120 seconds)')
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
   if args.loop_interval:
       if 30 <= args.loop_interval <= 120:
           config.main_loop_interval = args.loop_interval
       else:
           print("Loop interval must be between 30-120 seconds")
           sys.exit(1)
   
   # Create and run automation
   automation = TowerAutomation(config)
   
   try:
       automation.run()
   except Exception as e:
       logger.error(f"Fatal error: {e}", exc_info=True)
       sys.exit(1)

if __name__ == "__main__":
   main()
