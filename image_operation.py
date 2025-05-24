# image_operation.py - Enhanced with OCR for gem farming
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import logging
from pathlib import Path
import time  # Added missing import
import re

# Try to import pytesseract, but make it optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available, OCR features disabled")

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Result of template matching operation"""
    confidence: float
    location: Tuple[int, int]
    scale: float = 1.0
    method: str = "template"

class ImageMatcher:
    """Advanced image matching with multiple algorithms"""
    
    def __init__(self):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.template_cache = {}
        self._load_templates()
        
    def _load_templates(self):
        """Load all button templates"""
        template_dir = Path("templates")
        if not template_dir.exists():
            template_dir.mkdir(exist_ok=True)
            logger.warning("Templates directory created. Add button templates!")
            return
            
        # Load button templates - updated to match actual filenames
        button_names = [
            "5GemAdButton", "ClaimButton", "FinishAdButton", "FinishAdButton2",
            "IAgreeButton", "RewardGrantedButton", "close", "x_button",
            "spinning_gem", "start_wave", "retry",
            "workshop", "lab", "cards", "modules"
        ]
        
        for name in button_names:
            for ext in [".png", ".jpg"]:
                path = template_dir / f"{name}{ext}"
                if path.exists():
                    template = cv2.imread(str(path))
                    if template is not None:
                        self.template_cache[name] = template
                        logger.debug(f"Loaded template: {name}")
                    break
        
        # Also check state_images for button templates
        state_images_dir = Path("state_images")
        if state_images_dir.exists():
            for image_file in state_images_dir.glob("*Button.png"):
                name = image_file.stem
                template = cv2.imread(str(image_file))
                if template is not None:
                    self.template_cache[name] = template
                    logger.debug(f"Loaded state button template: {name}")
    
    def template_match(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.8) -> Optional[MatchResult]:
        """Multi-scale template matching"""
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            return None
            
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        best_match = None
        best_val = -1
        
        # Try multiple scales
        for scale in np.linspace(0.8, 1.2, 5):
            # Resize template
            width = int(gray_template.shape[1] * scale)
            height = int(gray_template.shape[0] * scale)
            scaled_template = cv2.resize(gray_template, (width, height))
            
            if (scaled_template.shape[0] > gray_image.shape[0] or 
                scaled_template.shape[1] > gray_image.shape[1]):
                continue
            
            # Match template
            result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val and max_val >= threshold:
                best_val = max_val
                # Calculate center position
                h, w = scaled_template.shape[:2]
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                
                best_match = MatchResult(
                    confidence=max_val,
                    location=(center_x, center_y),
                    scale=scale,
                    method="template"
                )
        
        return best_match
    
    def detect_ad_button(self, screen: np.ndarray) -> Optional[Tuple[int, int]]:
        """Specialized detection for 5-gem ad button"""
        # Multiple detection strategies for ad button
        
        # Strategy 1: Template matching
        for button_name in ["5GemAdButton", "5_gem_ad"]:
            if button_name in self.template_cache:
                result = self.template_match(screen, self.template_cache[button_name], threshold=0.7)
                if result:
                    return result.location
        
        # Strategy 2: Color-based detection for gem icon
        # Ad button typically has bright gem colors
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for gems (cyan/blue)
        lower_gem = np.array([80, 100, 100])
        upper_gem = np.array([100, 255, 255])
        
        # Create mask for gem colors
        gem_mask = cv2.inRange(hsv, lower_gem, upper_gem)
        
        # Focus on bottom 40% of screen where ad button appears
        h, w = screen.shape[:2]
        roi_mask = np.zeros_like(gem_mask)
        roi_mask[int(h*0.6):, int(w*0.2):int(w*0.8)] = 255
        
        # Combine masks
        combined_mask = cv2.bitwise_and(gem_mask, roi_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for button-sized contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 20000:  # Button size range
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it contains "5" text
                button_region = screen[y:y+h, x:x+w]
                if self._contains_five_text(button_region):
                    return (x + w//2, y + h//2)
        
        # Strategy 3: Look for video icon
        video_icon_pos = self._find_video_icon(screen)
        if video_icon_pos:
            return video_icon_pos
        
        return None
    
    def _contains_five_text(self, region: np.ndarray) -> bool:
        """Check if region contains '5' text"""
        if not TESSERACT_AVAILABLE:
            return False
            
        try:
            # Preprocess for OCR
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # OCR to find "5"
            text = pytesseract.image_to_string(binary, config='--psm 8 -c tessedit_char_whitelist=0123456789')
            return "5" in text
        except:
            return False
    
    def _find_video_icon(self, screen: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find video/play icon that might indicate ad button"""
        # This would use template matching for video icon
        if "video_icon" in self.template_cache:
            result = self.template_match(screen, self.template_cache["video_icon"], threshold=0.7)
            if result:
                return result.location
        return None

class OCRProcessor:
    """Text recognition for game values"""
    
    def __init__(self):
        # Precompile regex patterns
        self.number_pattern = re.compile(r'[\d,]+')
        self.wave_pattern = re.compile(r'Wave\s*(\d+)', re.IGNORECASE)
        
    def extract_number_from_region(self, region: np.ndarray, 
                                  preprocess: str = "thresh") -> Optional[int]:
        """Extract numeric value from region with preprocessing"""
        if not TESSERACT_AVAILABLE:
            return None
            
        try:
            # Preprocessing
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            if preprocess == "thresh":
                # Simple threshold
                _, processed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            elif preprocess == "adaptive":
                # Adaptive threshold for varying backgrounds
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
            elif preprocess == "invert":
                # Invert colors (white text on dark background)
                _, processed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            else:
                processed = gray
            
            # Denoise
            processed = cv2.medianBlur(processed, 3)
            
            # Scale up for better OCR
            scale_factor = 2
            width = int(processed.shape[1] * scale_factor)
            height = int(processed.shape[0] * scale_factor)
            processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # OCR with digit whitelist
            custom_config = '--psm 7 -c tessedit_char_whitelist=0123456789,.'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Clean and parse
            text = text.strip().replace(',', '').replace('.', '')
            numbers = self.number_pattern.findall(text)
            
            if numbers:
                return int(numbers[0])
                
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            
        return None
    
    def extract_text_from_region(self, region: np.ndarray) -> str:
        """Extract text from region"""
        if not TESSERACT_AVAILABLE:
            return ""
            
        try:
            # Preprocess
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = pytesseract.image_to_string(binary, config='--psm 7')
            return text.strip()
            
        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            return ""

# Global instances
image_matcher = ImageMatcher()
ocr_processor = OCRProcessor()

def find_button(screen: np.ndarray, button_name: str) -> Optional[Tuple[int, int]]:
    """Find a specific button on screen"""
    # Special handling for ad button
    if button_name in ["5_gem_ad", "5GemAdButton"]:
        return image_matcher.detect_ad_button(screen)
    
    # Check template cache
    if button_name in image_matcher.template_cache:
        template = image_matcher.template_cache[button_name]
        result = image_matcher.template_match(screen, template)
        if result and result.confidence > 0.75:
            return result.location
    
    # Fallback: try loading template directly
    template_path = Path(f"templates/{button_name}.png")
    if template_path.exists():
        template = cv2.imread(str(template_path))
        if template is not None:
            result = image_matcher.template_match(screen, template)
            if result and result.confidence > 0.75:
                return result.location
    
    return None

def extract_game_values(screen: np.ndarray) -> Dict[str, any]:
    """Extract numeric values from known screen regions"""
    values = {}
    
    # Define regions for different values (adjust based on your screen)
    # These are for 720x1280 resolution
    regions = {
        'gems': {
            'bbox': (50, 10, 200, 60),  # Top-left gem counter
            'preprocess': 'thresh',
            'fallback_bbox': (520, 10, 670, 60)  # Alternative position
        },
        'gold': {
            'bbox': (260, 10, 460, 60),  # Center-top gold counter
            'preprocess': 'thresh'
        },
        'wave': {
            'bbox': (500, 10, 650, 60),  # Top-right wave counter
            'preprocess': 'thresh',
            'text_region': True
        },
        'health': {
            'bbox': (300, 100, 420, 140),  # Health bar region
            'preprocess': 'adaptive'
        }
    }
    
    for name, config in regions.items():
        try:
            x1, y1, x2, y2 = config['bbox']
            region = screen[y1:y2, x1:x2]
            
            if config.get('text_region'):
                # Extract text first (for "Wave X" format)
                text = ocr_processor.extract_text_from_region(region)
                match = ocr_processor.wave_pattern.search(text)
                if match:
                    values[name] = int(match.group(1))
                else:
                    # Try numeric extraction
                    value = ocr_processor.extract_number_from_region(region, config['preprocess'])
                    if value is not None:
                        values[name] = value
            else:
                # Direct numeric extraction
                value = ocr_processor.extract_number_from_region(region, config['preprocess'])
                if value is not None:
                    values[name] = value
                elif 'fallback_bbox' in config:
                    # Try fallback position
                    x1, y1, x2, y2 = config['fallback_bbox']
                    region = screen[y1:y2, x1:x2]
                    value = ocr_processor.extract_number_from_region(region, config['preprocess'])
                    if value is not None:
                        values[name] = value
                        
        except Exception as e:
            logger.debug(f"Failed to extract {name}: {e}")
    
    # Log extracted values for debugging
    if values:
        logger.debug(f"Extracted values: {values}")
    
    return values

def detect_spinning_gem(screen: np.ndarray, tower_center: Tuple[int, int] = (360, 640),
                       orbit_radius: int = 200) -> Optional[Tuple[int, int]]:
    """Detect spinning gem around tower"""
    # Create orbital mask
    mask = np.zeros(screen.shape[:2], np.uint8)
    cv2.circle(mask, tower_center, orbit_radius + 50, 255, -1)
    cv2.circle(mask, tower_center, orbit_radius - 50, 0, -1)
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    
    # Spinning gem is typically bright cyan/blue
    lower_cyan = np.array([85, 150, 150])
    upper_cyan = np.array([95, 255, 255])
    
    # Alternative: bright white/yellow gems
    lower_bright = np.array([20, 100, 200])
    upper_bright = np.array([30, 255, 255])
    
    # Create color masks
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
    
    # Combine color masks
    color_mask = cv2.bitwise_or(cyan_mask, bright_mask)
    
    # Apply orbital mask
    gem_mask = cv2.bitwise_and(color_mask, mask)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    gem_mask = cv2.morphologyEx(gem_mask, cv2.MORPH_CLOSE, kernel)
    gem_mask = cv2.morphologyEx(gem_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(gem_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for gem-sized objects
    best_contour = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 3000:  # Gem size range
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Gems are typically circular
                if circularity > 0.6:
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Verify it's in orbital path
                        dist = np.sqrt((cx - tower_center[0])**2 + (cy - tower_center[1])**2)
                        orbital_score = 1.0 - abs(dist - orbit_radius) / orbit_radius
                        
                        if orbital_score > 0.7:
                            score = circularity * orbital_score
                            if score > best_score:
                                best_score = score
                                best_contour = (cx, cy)
    
    if best_contour:
        logger.debug(f"Spinning gem detected at {best_contour}")
        return best_contour
    
    return None

def find_ui_element(screen: np.ndarray, element_type: str) -> Optional[Tuple[int, int]]:
    """Find various UI elements by type"""
    # Common UI element positions (adjust for your screen)
    ui_positions = {
        'workshop_button': (100, 1200),
        'lab_button': (200, 1200),
        'cards_button': (300, 1200),
        'modules_button': (400, 1200),
        'settings_button': (650, 50),
        'back_button': (50, 50),
        'ultimate_golden_tower': (100, 600),
        'ultimate_black_hole': (620, 600),
    }
    
    # First try known positions
    if element_type in ui_positions:
        return ui_positions[element_type]
    
    # Then try template matching
    return find_button(screen, element_type)

def verify_screen_state(screen: np.ndarray, expected_state: str) -> bool:
    """Verify we're on the expected screen"""
    # Simple verification using key elements
    verifications = {
        'home': lambda s: find_button(s, 'start_wave') is not None,
        'in_wave': lambda s: extract_game_values(s).get('wave') is not None,
        'game_over': lambda s: find_button(s, 'retry') is not None,
        'workshop': lambda s: find_button(s, 'workshop_close') is not None,
        'watching_ad': lambda s: is_screen_mostly_black(s) or find_button(s, 'skip_ad') is not None,
    }
    
    if expected_state in verifications:
        return verifications[expected_state](screen)
    
    return False

def is_screen_mostly_black(screen: np.ndarray, threshold: float = 0.9) -> bool:
    """Check if screen is mostly black (ad playing)"""
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < 30)
    total_pixels = gray.size
    black_ratio = black_pixels / total_pixels
    return black_ratio > threshold

def save_debug_image(image: np.ndarray, name: str, directory: str = "debug"):
    """Save image for debugging"""
    debug_dir = Path(directory)
    debug_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = debug_dir / f"{name}_{timestamp}.png"
    
    cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logger.debug(f"Saved debug image: {filename}")
