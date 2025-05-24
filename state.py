# state.py - Updated with actual game states from your images
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import cv2
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class Region:
    """Defines a screen region for state detection"""
    x: int
    y: int
    width: int
    height: int
    weight: float = 1.0
    name: str = ""
    
    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_dict(self) -> dict:
        return {
            'x': self.x, 'y': self.y,
            'width': self.width, 'height': self.height,
            'weight': self.weight, 'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Region':
        return cls(**data)

@dataclass
class StateConfig:
    """Configuration for state detection"""
    min_confidence: float = 0.85
    histogram_bins: int = 16
    color_weight: float = 0.6
    edge_weight: float = 0.2
    structure_weight: float = 0.2
    use_gpu: bool = False
    cache_features: bool = True
    template_scale_range: Tuple[float, float] = (0.8, 1.2)
    template_scale_steps: int = 5

class FeatureCache:
    """Cache computed features for performance"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: any):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value

class State(ABC):
    def __init__(self, name: str, regions: List[Region], actions: Dict[str, Tuple[int, int]], 
                 config: StateConfig = None):
        self.name = name
        self.regions = regions
        self.actions = actions
        self.config = config or StateConfig()
        self.reference_image = None
        self.reference_features = {}
        self.feature_cache = FeatureCache() if config and config.cache_features else None
        
        # Initialize detectors
        self.sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04)
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2)
        
        # Load reference image
        self.load_reference_image()
        
    def load_reference_image(self):
        """Load and preprocess reference image"""
        image_path = Path(f'state_images/{self.name}.png')
        if not image_path.exists():
            logger.warning(f"No reference image found for {self.name}")
            return
            
        try:
            self.reference_image = cv2.imread(str(image_path))
            if self.reference_image is None:
                logger.error(f"Failed to load image: {image_path}")
                return
                
            # Pre-compute features for each region
            self._precompute_features()
            
        except Exception as e:
            logger.error(f"Failed to load reference image for {self.name}: {e}")
    
    def _precompute_features(self):
        """Pre-compute features for faster matching"""
        if self.reference_image is None:
            return
            
        for i, region in enumerate(self.regions):
            region_img = self._extract_region(self.reference_image, region)
            if region_img is None:
                continue
                
            # Compute multiple feature types
            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            
            # SIFT features
            kp_sift, desc_sift = self.sift.detectAndCompute(gray, None)
            
            # ORB features (faster alternative)
            kp_orb, desc_orb = self.orb.detectAndCompute(gray, None)
            
            # Color histogram
            hist = self._compute_color_histogram(region_img)
            
            # Edge map
            edges = cv2.Canny(gray, 50, 150)
            
            self.reference_features[i] = {
                'sift': (kp_sift, desc_sift),
                'orb': (kp_orb, desc_orb),
                'histogram': hist,
                'edges': edges,
                'gray': gray
            }
    
    def calculate_confidence(self, capture: np.ndarray) -> float:
        """Calculate state match confidence using multiple metrics"""
        if self.reference_image is None:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for i, region in enumerate(self.regions):
            # Check cache first
            cache_key = f"{self.name}_{i}_{capture.shape}"
            if self.feature_cache:
                cached_score = self.feature_cache.get(cache_key)
                if cached_score is not None:
                    total_score += cached_score * region.weight
                    total_weight += region.weight
                    continue
            
            # Extract region from capture
            cap_region = self._extract_region(capture, region)
            if cap_region is None or i not in self.reference_features:
                continue
                
            # Calculate multiple similarity metrics
            scores = {}
            
            # Color histogram similarity
            cap_hist = self._compute_color_histogram(cap_region)
            ref_hist = self.reference_features[i]['histogram']
            scores['color'] = cv2.compareHist(ref_hist, cap_hist, cv2.HISTCMP_CORREL)
            
            # Edge similarity
            cap_gray = cv2.cvtColor(cap_region, cv2.COLOR_BGR2GRAY)
            cap_edges = cv2.Canny(cap_gray, 50, 150)
            ref_edges = self.reference_features[i]['edges']
            scores['edge'] = self._edge_similarity(ref_edges, cap_edges)
            
            # Structural similarity (SSIM-like)
            scores['structure'] = self._structural_similarity(
                self.reference_features[i]['gray'], cap_gray
            )
            
            # Weighted combination
            region_score = (
                self.config.color_weight * scores['color'] +
                self.config.edge_weight * scores['edge'] +
                self.config.structure_weight * scores['structure']
            )
            
            # Cache the result
            if self.feature_cache:
                self.feature_cache.set(cache_key, region_score)
            
            total_score += region_score * region.weight
            total_weight += region.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _extract_region(self, image: np.ndarray, region: Region) -> Optional[np.ndarray]:
        """Safely extract region from image with bounds checking"""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = region.box
            
            # Clamp coordinates to image bounds
            x1 = int(np.clip(x1, 0, w))
            y1 = int(np.clip(y1, 0, h))
            x2 = int(np.clip(x2, 0, w))
            y2 = int(np.clip(y2, 0, h))
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            return image[y1:y2, x1:x2].copy()
            
        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
            return None
    
    def _compute_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute normalized color histogram"""
        # Calculate histogram for each channel
        hist_b = cv2.calcHist([image], [0], None, [self.config.histogram_bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [self.config.histogram_bins], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [self.config.histogram_bins], [0, 256])
        
        # Concatenate and normalize
        hist = np.concatenate([hist_b, hist_g, hist_r])
        cv2.normalize(hist, hist)
        
        return hist
    
    def _edge_similarity(self, edges1: np.ndarray, edges2: np.ndarray) -> float:
        """Calculate edge-based similarity"""
        if edges1.shape != edges2.shape:
            edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
        
        # Normalize edge maps
        edges1_norm = edges1.astype(np.float32) / 255.0
        edges2_norm = edges2.astype(np.float32) / 255.0
        
        # Calculate correlation
        correlation = np.corrcoef(edges1_norm.flatten(), edges2_norm.flatten())[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation):
            return 0.0
            
        return max(0.0, correlation)
    
    def _structural_similarity(self, gray1: np.ndarray, gray2: np.ndarray) -> float:
        """Calculate structural similarity index (SSIM)"""
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Constants for SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Convert to float
        img1 = gray1.astype(np.float64)
        img2 = gray2.astype(np.float64)
        
        # Calculate mean
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variance and covariance
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    @abstractmethod
    def execute_strategy(self, device, capture: np.ndarray, game_state: 'GameState'):
        """Execute state-specific strategy"""
        pass
    
    def get_action_point(self, action_name: str) -> Optional[Tuple[int, int]]:
        """Get action coordinates with validation"""
        return self.actions.get(action_name)
    
    def save_config(self, path: Path):
        """Save state configuration to JSON"""
        config = {
            'name': self.name,
            'regions': [r.to_dict() for r in self.regions],
            'actions': self.actions,
            'config': {
                'min_confidence': self.config.min_confidence,
                'histogram_bins': self.config.histogram_bins,
                'color_weight': self.config.color_weight,
                'edge_weight': self.config.edge_weight,
                'structure_weight': self.config.structure_weight,
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

class GameState:
    """Tracks overall game progress and statistics"""
    def __init__(self):
        self.wave_number = 0
        self.health = 100
        self.gold = 0
        self.gems = 0  # Added gem tracking
        self.last_state = None
        self.state_duration = 0
        self.total_runs = 0
        self.best_wave = 0
        self.current_run_start = None
        self.upgrade_counts = {}
        self.gems_collected = 0  # Total gems farmed
        self.ads_watched = 0  # Total ad gems claimed
        self.spinning_gems_collected = 0  # Total spinning gems
        self.in_wave = False  # Track if currently in wave
        
    def update(self, current_state: State):
        """Update game state tracking"""
        if current_state != self.last_state:
            self.state_duration = 0
            self.last_state = current_state
            
            # Track state transitions
            if isinstance(current_state, GameOverState):
                self.best_wave = max(self.best_wave, self.wave_number)
                self.total_runs += 1
                self.in_wave = False
            elif isinstance(current_state, (PlayAttackState, PlayDefenseState, PlayUtilityState)):
                self.in_wave = True
            elif isinstance(current_state, MenuState):
                self.in_wave = False
                
        else:
            self.state_duration += 1
    
    def track_upgrade(self, upgrade_type: str):
        """Track upgrade purchases"""
        self.upgrade_counts[upgrade_type] = self.upgrade_counts.get(upgrade_type, 0) + 1
    
    def track_gem_collection(self, amount: int, source: str = "unknown"):
        """Track gem collection"""
        self.gems += amount
        self.gems_collected += amount
        
        if source == "ad":
            self.ads_watched += 1
        elif source == "spinning":
            self.spinning_gems_collected += 1
        
        logger.info(f"💎 Collected {amount} gems from {source}! Total: {self.gems}")

# Concrete State Implementations based on your actual state images
class MenuState(State):
    def __init__(self):
        super().__init__(
            name="MenuState",
            regions=[
                Region(300, 750, 200, 100, weight=1.0, name="play_button"),
                Region(50, 50, 300, 150, weight=0.5, name="title"),
            ],
            actions={
                'play': (400, 800),
                'settings': (650, 50),
                'shop': (100, 1100),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        logger.info("In menu, starting new game")
        device.tap_point(self.get_action_point('play'))
        game_state.current_run_start = time.time()

class PlayAttackState(State):
    def __init__(self):
        super().__init__(
            name="PlayAttackState",
            regions=[
                Region(10, 780, 140, 40, weight=1.0, name="tab_indicator"),
                Region(250, 850, 200, 100, weight=0.7, name="damage_button"),
                Region(550, 850, 200, 100, weight=0.7, name="attack_speed_button"),
                Region(250, 950, 200, 100, weight=0.5, name="crit_chance_button"),
                Region(550, 950, 200, 100, weight=0.5, name="crit_damage_button"),
            ],
            actions={
                'damage': (350, 900),
                'attack_speed': (650, 900),
                'crit_chance': (350, 1000),
                'crit_damage': (650, 1000),
                'tab_defense': (200, 800),
                'tab_utility': (400, 800),
            }
        )
        self.upgrade_pattern = ['attack_speed', 'damage', 'crit_chance', 'crit_damage']
        self.upgrade_index = 0
    
    def execute_strategy(self, device, capture, game_state):
        # Adaptive upgrade strategy based on game progress
        if game_state.wave_number < 10:
            # Early game: focus on attack speed and damage
            priorities = ['attack_speed', 'damage']
        elif game_state.wave_number < 25:
            # Mid game: balanced upgrades
            priorities = self.upgrade_pattern
        else:
            # Late game: focus on crit for scaling
            priorities = ['crit_damage', 'crit_chance', 'damage', 'attack_speed']
        
        # Execute upgrade
        action = priorities[self.upgrade_index % len(priorities)]
        point = self.get_action_point(action)
        
        if point:
            device.tap_point(point)
            game_state.track_upgrade(action)
            logger.debug(f"Upgraded {action} (total: {game_state.upgrade_counts.get(action, 0)})")
        
        self.upgrade_index += 1
        
        # Tab switching logic
        if game_state.state_duration > 30:
            device.tap_point(self.get_action_point('tab_defense'))

class PlayDefenseState(State):
    def __init__(self):
        super().__init__(
            name="PlayDefenseState",
            regions=[
                Region(10, 780, 140, 40, weight=1.0, name="tab_indicator"),
                Region(250, 850, 200, 100, weight=0.8, name="health_button"),
                Region(550, 850, 200, 100, weight=0.6, name="regen_button"),
                Region(250, 950, 200, 100, weight=0.7, name="armor_button"),
            ],
            actions={
                'health': (350, 900),
                'regen': (650, 900),
                'armor': (350, 1000),
                'dodge': (650, 1000),
                'tab_utility': (400, 800),
                'tab_attack': (100, 800),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        # Health-based priority system
        if game_state.health < 30:
            # Emergency: prioritize immediate survival
            priorities = ['health', 'armor', 'regen']
        elif game_state.health < 60:
            # Low health: balanced defense
            priorities = ['armor', 'health', 'regen']
        else:
            # Healthy: invest in long-term survivability
            priorities = ['armor', 'dodge', 'regen', 'health']
        
        # Execute upgrade
        for priority in priorities:
            point = self.get_action_point(priority)
            if point:
                device.tap_point(point)
                game_state.track_upgrade(priority)
                break
        
        # Tab switching
        if game_state.state_duration > 20:
            device.tap_point(self.get_action_point('tab_utility'))

class PlayUtilityState(State):
    def __init__(self):
        super().__init__(
            name="PlayUtilityState",
            regions=[
                Region(10, 780, 140, 40, weight=1.0, name="tab_indicator"),
                Region(250, 850, 200, 100, weight=0.8, name="gold_bonus_button"),
                Region(550, 850, 200, 100, weight=0.7, name="exp_bonus_button"),
            ],
            actions={
                'gold_bonus': (350, 900),
                'exp_bonus': (650, 900),
                'cooldown_reduction': (350, 1000),
                'resource_efficiency': (650, 1000),
                'tab_attack': (100, 800),
                'tab_defense': (200, 800),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        # Economic strategy based on game phase
        if game_state.gold < 1000:
            # Poor: focus on gold generation
            device.tap_point(self.get_action_point('gold_bonus'))
            game_state.track_upgrade('gold_bonus')
        else:
            # Rich: invest in experience for long-term power
            device.tap_point(self.get_action_point('exp_bonus'))
            game_state.track_upgrade('exp_bonus')
        
        # Return to combat tabs
        if game_state.state_duration > 10:
            device.tap_point(self.get_action_point('tab_attack'))

class GameOverState(State):
    def __init__(self):
        super().__init__(
            name="GameOverState",
            regions=[
                Region(150, 980, 150, 60, weight=1.0, name="retry_button"),
                Region(250, 450, 300, 150, weight=0.8, name="game_over_text"),
                Region(450, 980, 150, 60, weight=0.5, name="menu_button"),
            ],
            actions={
                'retry': (225, 1010),
                'menu': (525, 1010),
                'share': (375, 1100),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        # Log performance
        logger.info(f"Game Over - Wave: {game_state.wave_number}, Best: {game_state.best_wave}")
        logger.info(f"Total upgrades: {sum(game_state.upgrade_counts.values())}")
        
        # Reset for new run
        game_state.wave_number = 0
        game_state.health = 100
        game_state.gold = 0
        game_state.upgrade_counts.clear()
        
        # Quick retry
        device.tap_point(self.get_action_point('retry'))

# GEM FARMING STATES - Updated to match actual filenames
class AdGemState(State):
    """Detects the 5GemAdButton state"""
    def __init__(self):
        super().__init__(
            name="5GemAdButton",  # Updated to match actual image name
            regions=[
                # The 5-gem ad button detection regions
                Region(250, 900, 220, 150, weight=1.0, name="ad_button"),
            ],
            actions={
                'tap_ad': (360, 975),
            },
            config=StateConfig(min_confidence=0.70)  # Lower threshold for ad detection
        )
        self.last_ad_time = 0
        self.ad_cooldown = 720  # 12 minutes
    
    def execute_strategy(self, device, capture, game_state):
        """Tap the ad button to start watching"""
        logger.info("📺 Found 5-gem ad button! Tapping...")
        device.tap_point(self.get_action_point('tap_ad'))
        self.last_ad_time = time.time()

class WatchingAdState(State):
    """State while watching an ad"""
    def __init__(self):
        super().__init__(
            name="WatchingAdState",  # Matches your image
            regions=[
                Region(0, 0, 720, 1280, weight=1.0, name="full_screen"),
            ],
            actions={
                'wait': (360, 640),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        """Just wait for ad to finish"""
        logger.debug("Watching ad...")
        time.sleep(1)  # Don't do anything while ad plays

class FinishedAdState(State):
    """State after ad finishes, need to claim reward"""
    def __init__(self):
        super().__init__(
            name="FinishedAdState",  # Matches your image
            regions=[
                Region(250, 750, 220, 100, weight=1.0, name="claim_area"),
            ],
            actions={
                'claim': (360, 800),
                'close': (650, 100),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        """Claim the gem reward"""
        logger.info("✅ Ad finished! Claiming reward...")
        
        # Try claim button first
        device.tap_point(self.get_action_point('claim'))
        time.sleep(1)
        
        # Then try close button
        device.tap_point(self.get_action_point('close'))
        
        # Track gem collection
        game_state.track_gem_collection(5, "ad")

class ClaimState(State):
    """State for claiming rewards"""
    def __init__(self):
        super().__init__(
            name="ClaimState",  # Matches your image
            regions=[
                Region(250, 750, 220, 100, weight=1.0, name="claim_button"),
            ],
            actions={
                'claim': (360, 800),
                'close': (650, 100),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        """Claim any reward"""
        logger.info("Claiming reward...")
        device.tap_point(self.get_action_point('claim'))
        time.sleep(0.5)
        device.tap_point(self.get_action_point('close'))

class EULAState(State):
    """EULA agreement screen"""
    def __init__(self):
        super().__init__(
            name="EULASTATE",  # Matches your image
            regions=[
                Region(200, 600, 320, 200, weight=1.0, name="eula_text"),
            ],
            actions={
                'agree': (360, 900),  # IAgreeButton location
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        """Accept EULA"""
        logger.info("Accepting EULA...")
        device.tap_point(self.get_action_point('agree'))

class ActiveBonusTimer(State):
    """State when bonus timer is active"""
    def __init__(self):
        super().__init__(
            name="ActiveBonusTimer",  # Matches your image
            regions=[
                Region(250, 450, 220, 100, weight=1.0, name="timer_display"),
            ],
            actions={
                'wait': (360, 640),
            }
        )
    
    def execute_strategy(self, device, capture, game_state):
        """Just wait or continue playing"""
        logger.debug("Bonus timer active")
        # Continue playing normally

class SpinningGemDetector:
    """Specialized detector for moving 2-gem diamond"""
    def __init__(self):
        self.tower_center = (360, 640)
        self.orbit_radius = 200
        self.last_gem_position = None
        self.last_detection_time = 0
        self.gem_cooldown = 30  # Gem respawns roughly every 30 seconds in wave
        
    def detect_spinning_gem(self, capture: np.ndarray, game_state: GameState) -> Optional[Tuple[int, int]]:
        """Detect the spinning 2-gem diamond"""
        if not game_state.in_wave:
            return None
            
        # Check cooldown
        if time.time() - self.last_detection_time < self.gem_cooldown:
            return None
        
        # Create mask for orbital area
        mask = np.zeros(capture.shape[:2], np.uint8)
        cv2.circle(mask, self.tower_center, self.orbit_radius + 50, 255, -1)
        cv2.circle(mask, self.tower_center, self.orbit_radius - 50, 0, -1)
        
        # Load gem template if available
        gem_template_path = Path('templates/spinning_gem.png')
        if not gem_template_path.exists():
            # Use color detection as fallback
            return self._detect_by_color(capture, mask)
        
        # Template matching
        gem_template = cv2.imread(str(gem_template_path))
        if gem_template is None:
            return self._detect_by_color(capture, mask)
        
        # Apply mask and search
        masked_capture = cv2.bitwise_and(capture, capture, mask=mask)
        
        # Try multiple scales
        best_match = None
        best_val = 0
        
        for scale in [0.8, 1.0, 1.2]:
            width = int(gem_template.shape[1] * scale)
            height = int(gem_template.shape[0] * scale)
            scaled_template = cv2.resize(gem_template, (width, height))
            
            result = cv2.matchTemplate(masked_capture, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val and max_val > 0.7:
                best_val = max_val
                h, w = scaled_template.shape[:2]
                best_match = (max_loc[0] + w//2, max_loc[1] + h//2)
        
        if best_match:
            self.last_gem_position = best_match
            self.last_detection_time = time.time()
            logger.info(f"🔹 Found spinning gem at {best_match}!")
            return best_match
        
        return None
    
    def _detect_by_color(self, capture: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """Fallback color-based detection for spinning gem"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(capture, cv2.COLOR_BGR2HSV)
        
        # Gem is typically cyan/blue color
        lower_cyan = np.array([80, 100, 100])
        upper_cyan = np.array([100, 255, 255])
        
        # Threshold the HSV image
        color_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Combine with orbital mask
        combined_mask = cv2.bitwise_and(color_mask, mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find gem-sized contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Gem size range
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Verify it's in orbital path
                    dist = np.sqrt((cx - self.tower_center[0])**2 + (cy - self.tower_center[1])**2)
                    if abs(dist - self.orbit_radius) < 50:
                        self.last_detection_time = time.time()
                        return (cx, cy)
        
        return None

class StateManager:
    """Enhanced state detection and management with gem focus"""
    def __init__(self, config_path: Optional[Path] = None):
        self.states = self._initialize_states()
        self.game_state = GameState()
        self.unknown_state_count = 0
        self.confidence_history = []
        self.state_transition_matrix = {}
        self.spinning_gem_detector = SpinningGemDetector()
        self.last_ad_check = 0
        self.ad_check_interval = 300  # 5 minutes
        
        if config_path and config_path.exists():
            self.load_config(config_path)
    
    def _initialize_states(self) -> List[State]:
        """Initialize all game states based on your actual images"""
        return [
            MenuState(),
            PlayAttackState(),
            PlayDefenseState(),
            PlayUtilityState(),
            GameOverState(),
            AdGemState(),          # 5GemAdButton
            WatchingAdState(),     # WatchingAdState
            FinishedAdState(),     # FinishedAdState
            ClaimState(),          # ClaimState
            EULAState(),          # EULASTATE
            ActiveBonusTimer(),   # ActiveBonusTimer
        ]
    
    def detect_state(self, capture: np.ndarray) -> Optional[State]:
        """Detect current state with confidence scoring"""
        # Convert PIL to numpy if needed
        if isinstance(capture, Image.Image):
            capture = np.array(capture)
        
        # Priority check for ad gem button
        current_time = time.time()
        if current_time - self.last_ad_check >= self.ad_check_interval:
            ad_state = next((s for s in self.states if isinstance(s, AdGemState)), None)
            if ad_state:
                ad_confidence = ad_state.calculate_confidence(capture)
                if ad_confidence >= ad_state.config.min_confidence:
                    logger.info(f"🎯 AD GEM DETECTED! Confidence: {ad_confidence:.2%}")
                    self.last_ad_check = current_time
                    return ad_state
        
        # Calculate confidence for each state
        confidences = {}
        for state in self.states:
            confidence = state.calculate_confidence(capture)
            confidences[state] = confidence
        
        # Find best match
        best_state = max(confidences, key=confidences.get)
        best_confidence = confidences[best_state]
        
        # Track confidence history
        self.confidence_history.append({
            'state': best_state.name,
            'confidence': best_confidence,
            'all_confidences': {s.name: c for s, c in confidences.items()}
        })
        
        # Keep only recent history
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        
        # Make decision based on confidence
        if best_confidence >= best_state.config.min_confidence:
            self.unknown_state_count = 0
            self._track_transition(best_state)
            return best_state
        else:
            self.unknown_state_count += 1
            
            # Use transition matrix to predict likely state
            if self.game_state.last_state and self.unknown_state_count < 3:
                predicted = self._predict_next_state(self.game_state.last_state)
                if predicted and confidences[predicted] > 0.6:
                    logger.info(f"Using predicted state: {predicted.name}")
                    return predicted
            
            logger.warning(
                f"Low confidence detection. Best: {best_state.name} ({best_confidence:.2f})"
            )
            
            # Return best guess if lost for too long
            if self.unknown_state_count > 5:
                return best_state
                
            return None
    
    def check_spinning_gem(self, device, capture: np.ndarray) -> bool:
        """Check for and collect spinning gem"""
        gem_position = self.spinning_gem_detector.detect_spinning_gem(capture, self.game_state)
        
        if gem_position:
            # Tap immediately with slight randomization
            tap_x = gem_position[0] + np.random.randint(-10, 10)
            tap_y = gem_position[1] + np.random.randint(-10, 10)
            
            device.tap_xy(tap_x, tap_y)
            
            # Track collection
            self.game_state.track_gem_collection(2, "spinning")
            
            return True
        
        return False
    
    def _track_transition(self, new_state: State):
        """Track state transitions for prediction"""
        if self.game_state.last_state:
            key = (self.game_state.last_state.name, new_state.name)
            self.state_transition_matrix[key] = \
                self.state_transition_matrix.get(key, 0) + 1
    
    def _predict_next_state(self, current_state: State) -> Optional[State]:
        """Predict most likely next state based on history"""
        transitions = {}
        
        for (from_state, to_state), count in self.state_transition_matrix.items():
            if from_state == current_state.name:
                transitions[to_state] = count
        
        if transitions:
            most_likely = max(transitions, key=transitions.get)
            for state in self.states:
                if state.name == most_likely:
                    return state
                    
        return None
    
    def execute_current_state(self, device, capture: np.ndarray):
        """Execute strategy for current state with gem priority"""
        # Always check for spinning gem first if in wave
        if self.game_state.in_wave:
            self.check_spinning_gem(device, capture)
        
        state = self.detect_state(capture)
        
        if state:
            self.game_state.update(state)
            state.execute_strategy(device, capture, self.game_state)
        else:
            logger.warning("Unknown state, waiting...")
    
    def save_config(self, path: Path):
        """Save state configurations and statistics"""
        config = {
            'states': {},
            'statistics': {
                'total_runs': self.game_state.total_runs,
                'best_wave': self.game_state.best_wave,
                'total_gems_collected': self.game_state.gems_collected,
                'ads_watched': self.game_state.ads_watched,
                'spinning_gems': self.game_state.spinning_gems_collected,
                'transition_matrix': {
                    f"{k[0]}->{k[1]}": v 
                    for k, v in self.state_transition_matrix.items()
                }
            }
        }
        
        # Save each state config
        for state in self.states:
            state_path = path.parent / f"{state.name}_config.json"
            state.save_config(state_path)
            config['states'][state.name] = str(state_path)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, path: Path):
        """Load saved configurations"""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            # Load statistics
            stats = config.get('statistics', {})
            self.game_state.total_runs = stats.get('total_runs', 0)
            self.game_state.best_wave = stats.get('best_wave', 0)
            self.game_state.gems_collected = stats.get('total_gems_collected', 0)
            self.game_state.ads_watched = stats.get('ads_watched', 0)
            self.game_state.spinning_gems_collected = stats.get('spinning_gems', 0)
            
            # Load transition matrix
            matrix = stats.get('transition_matrix', {})
            for key, value in matrix.items():
                from_state, to_state = key.split('->')
                self.state_transition_matrix[(from_state, to_state)] = value
                
            logger.info(f"Loaded configuration from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

def create_state_manager(config_path: Optional[Path] = None) -> StateManager:
    """Create and configure state manager with gem farming focus"""
    return StateManager(config_path)
