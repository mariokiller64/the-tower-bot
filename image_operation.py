# image_operation.py - Complete rewrite with advanced computer vision
import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

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
                best_match = MatchResult(
                    confidence=max_val,
                    location=max_loc,
                    scale=scale,
                    method="template"
                )
        
        return best_match
    
    def feature_match(self, image: np.ndarray, reference: np.ndarray,
                     min_matches: int = 10) -> Optional[MatchResult]:
        """Feature-based matching using SIFT"""
        try:
            # Detect features
            kp1, des1 = self.feature_detector.detectAndCompute(reference, None)
            kp2, des2 = self.feature_detector.detectAndCompute(image, None)
            
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return None
            
            # Match features
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) >= min_matches:
                # Calculate average position
                points = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                center = np.mean(points, axis=0)
                
                return MatchResult(
                    confidence=len(good_matches) / max(len(kp1), len(kp2)),
                    location=tuple(center.astype(int)),
                    method="feature"
                )
                
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            
        return None
    
    def histogram_match(self, region1: np.ndarray, region2: np.ndarray,
                       method: int = cv2.HISTCMP_CORREL) -> float:
        """Enhanced histogram comparison"""
        # Calculate histograms for each channel
        hist1_b = cv2.calcHist([region1], [0], None, [256], [0, 256])
        hist1_g = cv2.calcHist([region1], [1], None, [256], [0, 256])
        hist1_r = cv2.calcHist([region1], [2], None, [256], [0, 256])
        
        hist2_b = cv2.calcHist([region2], [0], None, [256], [0, 256])
        hist2_g = cv2.calcHist([region2], [1], None, [256], [0, 256])
        hist2_r = cv2.calcHist([region2], [2], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1_b, hist1_b)
        cv2.normalize(hist1_g, hist1_g)
        cv2.normalize(hist1_r, hist1_r)
        cv2.normalize(hist2_b, hist2_b)
        cv2.normalize(hist2_g, hist2_g)
        cv2.normalize(hist2_r, hist2_r)
        
        # Compare histograms
        score_b = cv2.compareHist(hist1_b, hist2_b, method)
        score_g = cv2.compareHist(hist1_g, hist2_g, method)
        score_r = cv2.compareHist(hist1_r, hist2_r, method)
        
        # Weighted average (green is most important for human perception)
        return 0.3 * score_b + 0.4 * score_g + 0.3 * score_r
    
    def structural_similarity(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """Calculate structural similarity index"""
        # Resize to same size if needed
        if region1.shape != region2.shape:
            region2 = cv2.resize(region2, (region1.shape[1], region1.shape[0]))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM-like metric
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return float(np.mean(ssim_map))

class OCRProcessor:
    """Text recognition for game values"""
    
    def __init__(self):
        self.digit_templates = self._load_digit_templates()
        
    def _load_digit_templates(self) -> dict:
        """Load pre-saved digit templates"""
        templates = {}
        # This would load actual digit templates in production
        return templates
    
    def extract_number(self, region: np.ndarray) -> Optional[int]:
        """Extract numeric value from region"""
        # Preprocess
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        digits = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > region.shape[0] * 0.5:  # Filter small noise
                digit_region = thresh[y:y+h, x:x+w]
                digit = self._recognize_digit(digit_region)
                if digit is not None:
                    digits.append(str(digit))
        
        if digits:
            return int(''.join(digits))
        return None
    
    def _recognize_digit(self, digit_region: np.ndarray) -> Optional[int]:
        """Recognize single digit using template matching"""
        # Simplified - would use actual template matching or ML model
        return None

# Global matcher instance
image_matcher = ImageMatcher()
ocr_processor = OCRProcessor()

def find_button(screen: np.ndarray, button_name: str) -> Optional[Tuple[int, int]]:
    """Find a specific button on screen"""
    # Load button template
    template_path = f"templates/buttons/{button_name}.png"
    try:
        template = cv2.imread(template_path)
        if template is None:
            return None
            
        result = image_matcher.template_match(screen, template)
        if result and result.confidence > 0.8:
            # Return center of button
            h, w = template.shape[:2]
            center_x = result.location[0] + w // 2
            center_y = result.location[1] + h // 2
            return (center_x, center_y)
            
    except Exception as e:
        logger.error(f"Button detection failed: {e}")
        
    return None

def extract_game_values(screen: np.ndarray) -> dict:
    """Extract numeric values from known screen regions"""
    values = {}
    
    # Define regions for different values
    regions = {
        'gold': (100, 50, 200, 80),
        'health': (300, 50, 400, 80),
        'wave': (500, 50, 600, 80),
    }
    
    for name, (x1, y1, x2, y2) in regions.items():
        region = screen[y1:y2, x1:x2]
        value = ocr_processor.extract_number(region)
        if value is not None:
            values[name] = value
            
    return values
