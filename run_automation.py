# run_automation.py - Gem farming focused automation with improved ad handling
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
from state import create_state_manager, StateManager, AdGemState, WatchingAdState, FinishedAdState
from image_operation import extract_game_values, find_button, detect_spinning_gem, is_screen_mostly_black

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
        self.main_loop_interval = 30  # Check every 30 seconds
        self.ad_check_interval = 300  # Check for ad every 5 minutes
        self.ad_retry_interval = 30  # Retry ad check every 30 seconds if no timer
        self.spinning_gem_check_interval = 15  # Check for spinning gem every 15 seconds in wave
        self.ad_watch_duration = 60  # 60 second ads
        self.ad_timeout = 120  # Max time to wait for ad to complete
        
        # Features
        self.enable_ocr = True
        self.gem_farming_mode = True
        self.auto_upgrades = True
        self.auto_wave_start = True
        self.aggressive_ad_mode = True  # Will go home/back to refresh ad timer
        
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
        self.ad_not_ready_count = 0
        
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
            'ad_not_ready_count': self.ad_not_ready_count,
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
        self.consecutive_ad_failures = 0
        
        # State tracking
        self.in_wave = False
        self.watching_ad = False
        self.ad_claim_attempts = 0
        
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
    
    def refresh_ad_timer(self) -> bool:
        """Go home and back to refresh ad timer"""
        logger.info("🔄 Refreshing ad timer by going home and back...")
        try:
            # Go to home screen
            self.device.home()
            time.sleep(2)
            
            # Return to app
            self.device.launch_app(self.config.app_package)
            time.sleep(3)
            
            # Wait for app to fully load
            for _ in range(10):
                if self.device.is_app_running(self.config.app_package):
                    time.sleep(2)  # Extra wait for UI to settle
                    return True
                time.sleep(1)
                
            return False
        except Exception as e:
            logger.error(f"Failed to refresh ad timer: {e}")
            return False
    
    def handle_ad_sequence(self) -> bool:
        """Complete ad watching sequence with better error handling"""
        logger.info("🔍 Starting ad sequence...")
        
        try:
            # Step 1: Take screenshot and check for ad button
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
                self.gem_stats.ad_not_ready_count += 1
                
                # If aggressive mode and no ad, try refresh
                if self.config.aggressive_ad_mode and self.consecutive_ad_failures < 3:
                    logger.info("⏳ Ad timer not ready, refreshing...")
                    if self.refresh_ad_timer():
                        time.sleep(2)
                        # Check again after refresh
                        capture = self.device.capture()
                        if capture:
                            ad_button = find_button(capture, "5_gem_ad")
                            if ad_button:
                                logger.info("✅ Ad button appeared after refresh!")
                            else:
                                self.consecutive_ad_failures += 1
                                return False
                
                return False
            
            # Reset failure counter on successful ad find
            self.consecutive_ad_failures = 0
            
            logger.info(f"📺 Found 5-gem ad button at {ad_button}!")
            self.gem_stats.ad_attempts += 1
            
            # Step 2: Tap ad button
            if self.config.randomize_taps:
                tap_x = ad_button[0] + np.random.randint(-10, 10)
                tap_y = ad_button[1] + np.random.randint(-10, 10)
            else:
                tap_x, tap_y = ad_button
            
            self.device.tap_xy(tap_x, tap_y)
            time.sleep(3)  # Wait for ad to start
            
            # Step 3: Watch ad with timeout
            self.watching_ad = True
            ad_start_time = time.time()
            logger.info(f"⏳ Watching ad...")
            
            # Monitor ad progress
            ad_finished = False
            while time.time() - ad_start_time < self.config.ad_timeout:
                # Check if ad finished
                capture = self.device.capture()
                if capture:
                    # Check for black screen (ad playing)
                    if is_screen_mostly_black(capture):
                        logger.debug("Ad is playing (black screen)")
                    else:
                        # Check for finish indicators
                        state = self.state_manager.detect_state(capture)
                        if isinstance(state, FinishedAdState):
                            logger.info("✅ Ad finished state detected!")
                            ad_finished = True
                            break
                        
                        # Check for reward/claim buttons
                        for button_name in ['claim', 'close', 'x_button', 'ClaimGem']:
                            if find_button(capture, button_name):
                                logger.info(f"✅ Found {button_name} button, ad finished!")
                                ad_finished = True
                                break
                
                # Show progress
                elapsed = int(time.time() - ad_start_time)
                if elapsed % 10 == 0:
                    logger.info(f"   {elapsed}s / {self.config.ad_watch_duration}s")
                
                time.sleep(1)
            
            self.watching_ad = False
            
            # Step 4: Claim reward with multiple attempts
            logger.info("💎 Attempting to claim reward...")
            self.ad_claim_attempts = 0
            
            # Wait a moment for claim screen to appear
            time.sleep(2)
            
            # Try multiple claim strategies
            claim_strategies = [
                # Strategy 1: Look for specific claim buttons
                lambda: self._try_claim_buttons(),
                # Strategy 2: Tap common positions
                lambda: self._try_claim_positions(),
                # Strategy 3: Use back button
                lambda: self._try_back_button(),
            ]
            
            for strategy in claim_strategies:
                if strategy():
                    break
                time.sleep(1)
            
            # Step 5: Verify gem increase
            time.sleep(3)
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
                    
                    # Try one more refresh to see if gems update
                    time.sleep(2)
                    capture = self.device.capture()
                    if capture:
                        values = extract_game_values(capture)
                        gems_after = values.get('gems', gems_before)
                        if gems_after > gems_before:
                            gem_gain = gems_after - gems_before
                            self.gem_stats.ads_watched += 1
                            logger.info(f"💎 SUCCESS (delayed)! Gained {gem_gain} gems!")
                            return True
            
        except Exception as e:
            logger.error(f"Ad sequence failed: {e}")
            self.gem_stats.ad_failures += 1
            self.watching_ad = False
        
        return False
    
    def _try_claim_buttons(self) -> bool:
        """Try to find and tap claim buttons"""
        capture = self.device.capture()
        if not capture:
            return False
        
        # Check for claim buttons
        button_names = ['claim', 'ClaimGem', 'Reward', 'close']
        for button_name in button_names:
            button_pos = find_button(capture, button_name)
            if button_pos:
                logger.info(f"Found {button_name} button at {button_pos}")
                self.device.tap_point(button_pos)
                time.sleep(1)
                return True
        
        return False
    
    def _try_claim_positions(self) -> bool:
        """Try common claim button positions"""
        claim_positions = [
            (360, 800),   # Center claim
            (360, 900),   # Lower center
            (360, 1000),  # Even lower
            (650, 100),   # Top-right X
            (50, 50),     # Top-left back
        ]
        
        for pos in claim_positions:
            logger.debug(f"Trying position {pos}")
            self.device.tap_point(pos)
            time.sleep(0.5)
        
        return True
    
    def _try_back_button(self) -> bool:
        """Try using device back button"""
        logger.debug("Trying back button")
        self.device.back()
        return True
    
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
                
                # Check if it's an ad state
                if isinstance(current_state, AdGemState):
                    # Found ad button, handle it immediately
                    self.handle_ad_sequence()
                elif isinstance(current_state, WatchingAdState):
                    # Currently watching ad, just wait
                    logger.debug("Currently watching ad...")
                elif isinstance(current_state, FinishedAdState):
                    # Ad finished, try to claim
                    logger.info("Ad finished, claiming reward...")
                    current_state.execute_strategy(self.device, capture, self.state_manager.game_state)
                else:
                    # Execute normal state strategy
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
        logger.info(f"⚙️ Aggressive ad mode: {self.config.aggressive_ad_mode}")
        
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
                
                # Priority 1: Ad gem check
                check_ad = False
                if not self.watching_ad:  # Don't check while watching ad
                    # Regular interval check
                    if current_time - self.last_ad_check >= self.config.ad_check_interval:
                        check_ad = True
                    # Aggressive retry if no recent success
                    elif (self.config.aggressive_ad_mode and 
                          current_time - self.last_ad_success > 900 and  # No success in 15 min
                          current_time - self.last_ad_check >= self.config.ad_retry_interval):
                        check_ad = True
                
                if check_ad:
                    self.handle_ad_sequence()
                    self.last_ad_check = current_time
                
                # Priority 2: Spinning gem check (every 15 seconds if in wave)
                if self.in_wave and current_time - self.last_spinning_check >= self.config.spinning_gem_check_interval:
                    self.check_spinning_gem()
                    self.last_spinning_check = current_time
                
                # Regular automation cycle
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
        print("  ad     - Force ad check now")
        print("  refresh - Refresh ad timer")
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
                elif cmd == "ad":
                    print("Forcing ad check...")
                    self.handle_ad_sequence()
                elif cmd == "refresh":
                    print("Refreshing ad timer...")
                    self.refresh_ad_timer()
                elif cmd == "stop":
                    self.stop()
                elif cmd == "help":
                    print("\nCommands: stats, freq, ad, refresh, stop, help\n")
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
        print(f"⏳ Ad Timer Not Ready: {stats['ad_not_ready_count']} times")
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
        print("Enter new frequency (10-120 seconds):")
        
        try:
            new_freq = int(input("New frequency: "))
            if 10 <= new_freq <= 120:
                self.config.main_loop_interval = new_freq
                print(f"✅ Check frequency set to {new_freq} seconds")
            else:
                print("❌ Frequency must be between 10-120 seconds")
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
    parser.add_argument('--loop-interval', type=int, help='Main loop interval (10-120 seconds)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--no-aggressive', action='store_true',
                        help='Disable aggressive ad mode')
    
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
        if 10 <= args.loop_interval <= 120:
            config.main_loop_interval = args.loop_interval
        else:
            print("Loop interval must be between 10-120 seconds")
            sys.exit(1)
    if args.no_aggressive:
        config.aggressive_ad_mode = False
    
    # Create and run automation
    automation = TowerAutomation(config)
    
    try:
        automation.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
