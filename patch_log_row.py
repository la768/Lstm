import os
import time
import csv
import pyautogui
import pytesseract
from PIL import Image
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from hmmlearn.hmm import GaussianHMM
import joblib
from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
from lightgbm import LGBMClassifier, early_stopping as lgbm_early_stopping
from sklearn.semi_supervised import LabelPropagation
import warnings
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from rapidocr_onnxruntime import RapidOCR
import easyocr
_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA

try:
    from agent_ai import MultiAgentSystem
except Exception as exc:
    MultiAgentSystem = None
    print(f"[MULTI-AI] DISABLED - could not import agent_ai.py: {exc}")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
 

GLOBAL_DATA_WINDOW = 30      
MODEL_WINDOW =  30
MIN_ROWS_FOR_TRAINING = 30            
MODEL_ACCURACY_THRESHOLD = 0.50  # Min accuracy to allow model to vote
CONFIDENCE_THRESHOLD = 0.15   # Skip if consensus confidence below this
WARMUP_ROUNDS = 1             # Deprecated now — replaced by FILTER_AFTER_PREDICTIONS
FILTER_AFTER_PREDICTIONS = 500 # Only start filtering models after this many predictions
RETRAIN_INTERVAL = 60
  

REGION_PERIOD = (1170, 322, 190, 69)
REGION_BIGSMALL = (1480, 322, 100, 69)
REGION_FULL_PERIOD = (1155, 300, 215, 750)

REGION_FULL_BIGSMALL = (1430, 289, 195, 730)

REGION_FULL_NUMBERS = (1320, 289, 175, 715)
REGION_NUMBER = (1320, 289, 175, 100)   
REGION_2PERIOD = (1170, 222, 190, 230)
REGION_2BIGSMALL = (1480, 222, 100, 230)
 
BASE_FOLDER = r"C:\Users\User\Desktop\APPS\AI_US\Lstm\Ai_folder"
SCREENSHOT_FOLDER = os.path.join(BASE_FOLDER, "screenshots")
CSV_PATH = os.path.join(BASE_FOLDER, "ocr_result2.csv")
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True) 
rapid_ocr = RapidOCR()

NUMBER_TO_LETTER = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}
LETTER_TO_NUMBER = {v:k for k,v in NUMBER_TO_LETTER.items()}

COLOR_MAP = {
    "red": 4,
    "green": 3,
    "red & purple": 5,
    "green & purple": 6,
 
}

COLOR_TO_NUMBERS = {
    4: [ 2, 4, 6, 8],     # red → even numbers
    3: [1, 3, 7, 9],         # green → odd numbers (except 5)
    5: [0],                   # red & purple → only 0
    6: [5],                   # green & purple → only 5
}

# Decode color for display
COLOR_DECODE = {4: "red", 3: "green", 5: "red & purple", 6: "green & purple"}

def number_to_bigsmall(num): 
    return 0 if num <= 4 else 1
 
short_names = {
    "KNN": "KNN", "HMM": "HMM", "LGBM": "LGBM", "ET": "ET",
    "PA": "PA", "SVM": "SVM",
    "GPC": "GPC", "RegimeTrend": "RT", "RIDGE": "RG", "ADA": "ADA",
    "HGB": "HGB", "LR": "LR", "RF": "RF",
    "CatBoost": "CAT", "Markov": "Mkv", "NGRAM": "NGRM",
    "BAG": "BAG", "LBLPROP": "LBL", "VOTER": "VTR"
}
 
last_trigger_time = None
last_logged_period = None
banner_printed = False
last_X_used = None
total_predictions = 0
correct_predictions = 0
# Combined AI system accuracy tracking
ai_total_predictions = 0
ai_correct_predictions = 0
last_ai_prediction_bs = None
last_prediction_bs = None
last_prediction_num = None
last_prediction_col = None
BOOTSTRAP_MODE = True
existing_periods = set()
rounds_since_retrain = 0

MODELS = {}
HGB_WINDOW_MODEL = None
scaler = None

model_stats_bs = defaultdict(lambda: {"correct": 0, "total": 0, "recent_correct": 0, "recent_total": 0, "recent_results": []})
model_stats_num = defaultdict(lambda: {"correct": 0, "total": 0, "recent_correct": 0, "recent_total": 0, "recent_results": []})
model_stats_col = defaultdict(lambda: {"correct": 0, "total": 0, "recent_correct": 0, "recent_total": 0, "recent_results": []})

last_model_predictions_bs = {}
last_model_predictions_num = {}
last_model_predictions_col = {}

 
def encode_bs(x):
    value = str(x).strip().lower()
    if value == "big":
        return 1
    if value == "small":
        return 0
    raise ValueError(f"Invalid BigSmall label: {x!r}")

def decode_bs(x):
    if x is None:
        return "NONE"
    return "BIG" if x == 1 else "SMALL"

def encode_num(x):
    if isinstance(x, str):
        x = x.strip()
    try:
        value = int(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number: {x!r}") from exc
    if value not in NUMBER_TO_LETTER:
        raise ValueError(f"Number must be a single digit 0-9, got {x!r}")
    return NUMBER_TO_LETTER[value]

def decode_num(x):
    if isinstance(x, (int, np.integer)):
        value = int(x)
    else:
        value = LETTER_TO_NUMBER.get(str(x).strip().upper())
    if value not in range(10):
        raise ValueError(f"Invalid encoded number: {x!r}")
    return value

def encode_col(x):
    """Convert color string to encoded int (3-6)"""
    value = str(x).strip().lower()
    if value not in COLOR_MAP:
        raise ValueError(f"Invalid color: {x!r}")
    return COLOR_MAP[value]

def ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Period", "Numbers", "BigSmall", "Color"])

 
def ocr_period(img):
    cv_img = np.array(img.convert("L"))
    resized = cv2.resize(cv_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_pil = Image.fromarray(thresh)
    return pytesseract.image_to_string(
        img_pil,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()

def ocr_bigsmall(img):
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > 160 else 0)
    img_np = np.array(img)
    results, elapse = rapid_ocr(img_np)
    if results:
        text = " ".join([r[1].strip().lower() for r in results])
        if "big" in text: return "big"
        if "small" in text: return "small"
    return "unknown"

def ocr_numbers_split_dual(n_img_full):
 
    cv_img = np.array(n_img_full)
    
    if len(cv_img.shape) == 2:
        bgr = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv_img.copy()
    
    h, w = bgr.shape[:2]
    mid = h // 2
    
    # Split into top and bottom halves
    top_half = bgr[0:mid, :]
    bottom_half = bgr[mid:h, :]
    
    # Run the exact same OCR pipeline on each half
    top_digit = _ocr_single_half(top_half)
    bottom_digit = _ocr_single_half(bottom_half)
    
    return top_digit, bottom_digit


def _ocr_single_half(half_bgr):
    """
    Process a single image half through the same color-masking + EasyOCR pipeline
    that your ocr_numbers() function uses.
    """
    gray = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2HSV)
    
    # Color masking (copied from your ocr_numbers)
    red1 = cv2.inRange(hsv, (0, 30, 30), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 30, 30), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    green = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    purple = cv2.inRange(hsv, (125, 30, 30), (165, 255, 255))
    
    mask = cv2.bitwise_or(red, green)
    mask = cv2.bitwise_or(mask, purple)
    
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    
    if not contours:
        return ""
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w_c, h_c = cv2.boundingRect(largest)
    
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w_c + pad)
    y2 = min(gray.shape[0], y + h_c + pad)
    
    roi_gray = gray[y1:y2, x1:x2]
    roi_bin = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel = np.ones((2, 2), np.uint8)
    roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel)
    roi_bin = cv2.resize(roi_bin, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # EasyOCR
    results = _reader.readtext(roi_bin, allowlist='0123456789')
    
    if results:
        for bbox, text, conf in results:
            digits = ''.join(filter(str.isdigit, text))
            if len(digits) == 1:
                return digits
    
    # Inverted fallback
    roi_inv = cv2.bitwise_not(roi_bin)
    results = _reader.readtext(roi_inv, allowlist='0123456789')
    if results:
        for bbox, text, conf in results:
            digits = ''.join(filter(str.isdigit, text))
            if len(digits) == 1:
                return digits
    
    return ""


def ocr_numbers(img):
  
    results, elapse = rapid_ocr(img)
    
    
    if not results:
        return ""
    
    # Sort by confidence descending, take the best
    results.sort(key=lambda x: x[2], reverse=True)
    
    for box, text, conf in results:
        cleaned = text.strip()
        # Extract only digits
        digits = ''.join(filter(str.isdigit, cleaned))

        if len(digits) == 1: 
            return digits
     
    return ""

def get_color_from_number(num_str):
    value = str(num_str).strip()
    if len(value) != 1 or not value.isdigit():
        raise ValueError(f"Invalid input: '{num_str}' is not a valid number")

    last_digit = int(value)
    color_map = {
        0: "red & purple", 1: "green", 2: "red", 3: "green",
        4: "red", 5: "green & purple", 6: "red", 7: "green",
        8: "red", 9: "green",
    }
    return color_map.get(last_digit)


def normalize_digit(value):
    """Return a validated single digit, or raise for malformed OCR/data."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) != 1 or not text.isdigit():
        raise ValueError(f"Expected one digit 0-9, got {value!r}")
    return text


def _model_classes(model):
    """Return classifier classes for a plain estimator or a Pipeline."""
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return classes
    steps = getattr(model, "named_steps", None)
    if steps:
        return getattr(list(steps.values())[-1], "classes_", None)
    return None


def load_clean_dataframe(csv_file):
    """Load only usable rows and derive color from the validated digit.

    The color is deterministic for a Wingo digit, so deriving it here prevents
    one bad OCR color from becoming a separate training label.
    """
    df = pd.read_csv(csv_file)
    required = {"Numbers", "BigSmall", "Color"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    original_len = len(df)
    df = df.copy()
    df["BigSmall"] = df["BigSmall"].astype(str).str.strip().str.lower()
    df["Numbers"] = pd.to_numeric(
        df["Numbers"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
        errors="coerce",
    )
    valid_numbers = df["Numbers"].notna() & df["Numbers"].between(0, 9)
    valid_numbers &= df["Numbers"].mod(1).eq(0)
    valid_bs = df["BigSmall"].isin(["big", "small"])
    valid_colors = df["Color"].astype(str).str.strip().str.lower().isin(COLOR_MAP)
    df = df[valid_numbers & valid_bs & valid_colors].copy()
    df["Numbers"] = df["Numbers"].astype(int)
    df["Color"] = df["Numbers"].map(get_color_from_number)

    dropped = original_len - len(df)
    if dropped:
        print(f"⚠️ Ignored {dropped} invalid CSV row(s); using {len(df)} clean rows")
    return df

 
def wait_for_capture_time():
    global last_trigger_time
    while True:
        now = datetime.now()
        if now.second in (3, 33):
            key = now.strftime("%H:%M:%S")
            if key != last_trigger_time:
                last_trigger_time = key
                return
        time.sleep(0.05)

class HMMWrapper:
            def __init__(self, hmm_models, classes):
                self.hmm_models = hmm_models
                self.classes = classes
            
            def predict(self, X):
                preds = []
                for i in range(len(X)):
                    sample = X[i:i+1]
                    scores = {}
                    for c, model in self.hmm_models.items():
                        try:
                            scores[c] = model.score(sample, lengths=[1])
                        except:
                            scores[c] = -float('inf')
                    preds.append(max(scores, key=scores.get) if scores else self.classes[0])
                return np.array(preds)

def bootstrap_capture_all_rows():
    global last_logged_period
    EXPECTED_ROWS = 10
    MAX_RETRIES = 37
    RETRY_DELAY = 3

    print("\n🔄 BOOTSTRAP MODE: Capturing full visible history...")
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n📸 Attempt {attempt}/{MAX_RETRIES}")

        # ── Screenshots (unchanged) ──
        p_img = pyautogui.screenshot(region=REGION_FULL_PERIOD)
        b_img = pyautogui.screenshot(region=REGION_FULL_BIGSMALL)
        n_img = pyautogui.screenshot(region=REGION_FULL_NUMBERS)

        # ── Period OCR (unchanged) ──
        p_text = pytesseract.image_to_string(
            p_img.convert("L"),
            config="--psm 6 -c tessedit_char_whitelist=0123456789"
        )

        # ── Big/Small OCR (unchanged) ──
        b_text = pytesseract.image_to_string(
            b_img.convert("L"),
            config="--psm 6"
        ).lower() 
        numbers_raw, elapse = rapid_ocr(n_img)
        
        if numbers_raw:
            numbers_raw.sort(key=lambda x: x[0][0][1])  # sort top-to-bottom
            all_numbers = []
            for box, text, conf in numbers_raw:
                cleaned = text.strip()
                cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.')
                if cleaned:
                    try:
                        cleaned = str(int(float(cleaned)))
                    except:
                        pass
                    all_numbers.append(cleaned)
        else:
            all_numbers = []

        top_numbers = all_numbers[:5]
        bot_numbers = all_numbers[5:10]

        print(f"📊 Top half OCR: {repr('\\n'.join(top_numbers) if top_numbers else '')}")
        print(f"📊 Bottom half OCR: {repr('\\n'.join(bot_numbers) if bot_numbers else '')}")
        print(f"🔢 Top half parsed: {top_numbers}")
        print(f"🔢 Bottom half parsed: {bot_numbers}")

        # ── Parse periods (unchanged) ──
        period_lines = p_text.split("\n")
        periods_found = []
        for p_line in period_lines:
            digits = ''.join(c for c in p_line if c.isdigit())
            if len(digits) >= 3:
                periods_found.append(digits[-5:])

        # ── Parse big/small (unchanged) ──
        bs_lines = b_text.split("\n")
        bs_found = []
        for b_line in bs_lines:
            bl = b_line.strip().lower()
            if "big" in bl:
                bs_found.append("big")
            elif "small" in bl:
                bs_found.append("small")

        # ── Build the full stack (unchanged logic) ──
        stack = []
        empty_number_count = 0
        total_rows = len(periods_found)

        for i in range(total_rows):
            period = periods_found[i]
            if i >= len(bs_found):
                break

            if i < 5:
                idx_in_half = i
                num = top_numbers[idx_in_half] if idx_in_half < len(top_numbers) else ""
            else:
                idx_in_half = i - 5
                num = bot_numbers[idx_in_half] if idx_in_half < len(bot_numbers) else ""

            if not num:
                empty_number_count += 1
                print(f"  ⚠️ Empty number field for period {period} (half index {idx_in_half})")

            if len(str(num).strip()) != 1 or not str(num).strip().isdigit():
                continue
            color = get_color_from_number(num)
            stack.append((period, num, bs_found[i], color))

        print(f"🔍 Detected {len(stack)} valid rows ({empty_number_count} with empty numbers)")

        if len(stack) >= EXPECTED_ROWS and empty_number_count == 0:
            print("✅ Required rows detected with all numbers populated. Proceeding...")
            break
        else:
            if empty_number_count > 0:
                print(f"⚠️ {empty_number_count} row(s) have empty number fields. Retrying in {RETRY_DELAY}s...")
            elif len(stack) < EXPECTED_ROWS:
                print(f"⚠️ Only {len(stack)} rows found (need {EXPECTED_ROWS}). Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    else:
        print("❌ Max retries reached. Proceeding with best available data...")

    # Filter and log (unchanged)
    valid_stack = [(p, num, b, c) for p, num, b, c in stack if num]
    skipped = len(stack) - len(valid_stack)
    if skipped:
        print(f"⚠️ Skipping {skipped} row(s) with empty numbers")

    added = 0
    while valid_stack:
        p, num, b, c = valid_stack.pop()
        if log_row(p, num, b, c):
            added += 1
    print(f"📥 Bootstrapped {added} rows in chronological order\n")

def capture_post_bootstrap_rows():
    global existing_periods
    
    p_img = pyautogui.screenshot(region=REGION_2PERIOD)
    b_img = pyautogui.screenshot(region=REGION_2BIGSMALL)
    n_img = pyautogui.screenshot(region=REGION_FULL_NUMBERS)

    p_text = pytesseract.image_to_string(
        p_img.convert("L"),
        config="--psm 6 -c tessedit_char_whitelist=0123456789"
    )
    b_text = pytesseract.image_to_string(
        b_img.convert("L"),
        config="--psm 6"
    ).lower()
    n_text = pytesseract.image_to_string(
        n_img.convert("L"),
        config="--psm 6 -c tessedit_char_whitelist=0123456789"
    )

    period_lines = p_text.split("\n")
    bs_lines = b_text.split("\n")
    n_lines = n_text.split("\n")

    stack = []
    for p_line, b_line, n_line in zip(period_lines, bs_lines, n_lines):
        digits = ''.join(c for c in p_line if c.isdigit())
        if len(digits) >= 3 and ("big" in b_line or "small" in b_line):
            period = digits[-5:]
            bs = "big" if "big" in b_line else "small"
            if period in existing_periods:
                continue
            num_digits = ''.join(c for c in n_line if c.isdigit())
            color = get_color_from_number(num_digits) if num_digits else ""
            stack.append((period, num_digits, bs, color))

    added = 0
    while stack:
        p, num, b, c = stack.pop()
        if log_row(p, num, b, c):
            added += 1


def capture_row(max_retries=5, retry_delay=4):
    global last_logged_period
    for attempt in range(1, max_retries + 1):
        p_img = pyautogui.screenshot(region=REGION_PERIOD)
        b_img = pyautogui.screenshot(region=REGION_BIGSMALL)
        n_img = pyautogui.screenshot(region=REGION_NUMBER)

        period = ocr_period(p_img)
        bigsmall = ocr_bigsmall(b_img)
        numbers = ocr_numbers(n_img)

        digits = ''.join(c for c in period if c.isdigit())
        if len(digits) >= 3:
            period = digits[-5:]
            bigsmall = bigsmall.lower()
            if bigsmall not in ("big", "small"):
                print(f"⛔ Invalid BIG/SMALL OCR (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            if not numbers or not numbers.strip().isdigit():
                print(f"⛔ Invalid number OCR (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            if period == last_logged_period:
                print(f"⏳ Same period {period} (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            color = get_color_from_number(numbers)
            return period, numbers, bigsmall, color

        print(f"⛔ Invalid OCR (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
        time.sleep(retry_delay)

    return None, None, None, None


def log_row(period, numbers, bigsmall, color):
    global last_logged_period, existing_periods
    try:
        numbers = normalize_digit(numbers)
        bigsmall = str(bigsmall).strip().lower()
        encode_bs(bigsmall)
        expected_color = get_color_from_number(numbers)
    except ValueError as exc:
        print(f"⛔ Row rejected: {exc}")
        return False

    supplied_color = str(color).strip().lower()
    if supplied_color != expected_color:
        print(f"⚠️ Correcting inconsistent color for {numbers}: "
              f"{supplied_color!r} → {expected_color!r}")
    color = expected_color

    if period in existing_periods:
        print(f"⚠️ Duplicate skipped: {period}")
        return False
    
    try:
        recent_check = 300
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-recent_check:]:
                parts = line.strip().split(',')
                if len(parts) >= 1 and parts[0] == str(period):
                    existing_periods.add(period)
                    print(f"⚠️ Duplicate skipped (tail check): {period}")
                    return False
    except Exception:
        pass

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([period, numbers, bigsmall, color])

    last_logged_period = period
    existing_periods.add(period)
    print(f"\n📝 Logged → Period: {period}, Numbers: {numbers}, BigSmall: {bigsmall}, Color: {color}")
    return True

def build_features_bs(seq, window=MODEL_WINDOW):
    X, y = [], []
    for i in range(window, len(seq)):
        X.append(seq[i-window:i])
        y.append(seq[i])
    return np.array(X), np.array(y)


def build_combined_features(bs_seq, num_seq, col_seq, window=MODEL_WINDOW):
   
    num_ints = [decode_num(x) if isinstance(x, str) else x for x in num_seq]
    
    X_all, y_bs, y_num, y_col = [], [], [], []
    
    for i in range(window, min(len(bs_seq), len(num_ints), len(col_seq))):
        feat = []
        feat.extend(bs_seq[i-window:i])   # past BigSmall
        feat.extend(num_ints[i-window:i])  # past Numbers
        feat.extend(col_seq[i-window:i])   # past Colors
        X_all.append(feat)
        y_bs.append(bs_seq[i])
        y_num.append(num_ints[i])
        y_col.append(col_seq[i])
    
    return np.array(X_all), np.array(y_bs), np.array(y_num), np.array(y_col)


def create_hgb_window_features(window):
    window = np.array(window)
    features = []
    features.append(np.mean(window[-5:]))
    features.append(np.mean(window[-10:]))
    features.append(np.mean(window))
    features.append(np.std(window[-5:]))
    features.append(np.std(window[-10:]))
    features.append(window[-1])
    features.append(window[-1] - window[-2])
    features.append(np.sum(window[-5:]))
    features.append(np.sum(window[-10:]))
    last10 = window[-10:]
    flip_count = np.sum(last10[1:] != last10[:-1])
    features.append(flip_count)
    return np.array(features)

def markov_predict(seq):
    if len(seq) < 2:
        return None
    transitions = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
    for i in range(len(seq)-1):
        transitions[(seq[i], seq[i+1])] += 1
    last = seq[-1]
    next0 = transitions[(last, 0)]
    next1 = transitions[(last, 1)]
    return 1 if next1 > next0 else 0


def predict_regime_trend(seq, window=20):
    if len(seq) < window:
        return None
    recent = seq[-window:]
    ones = sum(recent)
    zeros = window - ones
    if ones / window >= 0.60:
        return 1
    if zeros / window >= 0.60:
        return 0
    return 1 - recent[-1]


def ngram_predict(seq, n=3):
    if len(seq) < n + 1:
        return None
    counts = defaultdict(lambda: [0, 0])
    for i in range(len(seq) - n):
        pattern = tuple(seq[i:i+n])
        next_val = seq[i+n]
        counts[pattern][next_val] += 1
    last_pattern = tuple(seq[-n:])
    if last_pattern in counts:
        return 1 if counts[last_pattern][1] > counts[last_pattern][0] else 0
    return None


def ngram_predict_num(seq, n=3):
    if len(seq) < n + 1:
        return None, 0.0
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(seq) - n):
        pattern = tuple(seq[i:i+n])
        next_val = seq[i+n]
        counts[pattern][next_val] += 1
    last_pattern = tuple(seq[-n:])
    if last_pattern in counts:
        total = sum(counts[last_pattern].values())
        best_val = max(counts[last_pattern], key=counts[last_pattern].get)
        return best_val, counts[last_pattern][best_val] / total
    return None, 0.0


def ngram_predict_col(seq, n=3):
    if len(seq) < n + 1:
        return None
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(seq) - n):
        pattern = tuple(seq[i:i+n])
        next_val = seq[i+n]
        counts[pattern][next_val] += 1
    last_pattern = tuple(seq[-n:])
    if last_pattern in counts:
        best_val = max(counts[last_pattern], key=counts[last_pattern].get)
        return best_val
    return None

 
def train_hgb_window(seq, window=20):
    X, y = [], []
    for i in range(window, len(seq)-1):
        sub = seq[i-window:i]
        target = seq[i]
        feats = create_hgb_window_features(sub)
        X.append(feats)
        y.append(target)
    if len(X) < 10:
        return None
    X = np.array(X)
    y = np.array(y)
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05)
    model.fit(X, y)
    return model


def _train_single_classifier(model_class, kwargs, X_train, y_train, X_val, y_val, name, use_scaled=False):
 
    try:
        model = model_class(**kwargs).fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            acc = model.score(X_val, y_val)
        else:
            preds = model.predict(X_val)
            acc = np.mean(preds == y_val)
        return model, acc
    except Exception as e:
        print(f"  ⚠️ {name} failed: {e}")
        return None, 0.0


def train_models(csv_file):
    df = load_clean_dataframe(csv_file)
    
    if len(df) < 10:
        print(f"⚠️ Only {len(df)} rows, need at least 10. Skipping model training.")
        return
    
    # Encode
    bs_seq = df["BigSmall"].apply(encode_bs).tolist()
    num_seq = df["Numbers"].apply(encode_num).tolist()
    col_seq = df["Color"].apply(encode_col).tolist()
    
    # Build combined features
    X_combined, y_bs, y_num, y_col = build_combined_features(bs_seq, num_seq, col_seq, window=MODEL_WINDOW)
    
    if X_combined.size == 0 or len(X_combined) < 10:
        print("⚠️ Not enough combined features. Skipping model training.")
        return
    
    unique_bs = set(y_bs)
    if len(unique_bs) < 2:
        print(f"⚠️ Only 1 BigSmall class ({unique_bs}), skipping.")
        return
    
    # ── Stratified split to preserve class distribution ──
    X_train, X_val, y_bs_train, y_bs_val = train_test_split(
        X_combined, y_bs, test_size=0.2, shuffle=False  # chronological, no shuffle
    )
    # For num and col, align indices
    _, _, y_num_train, y_num_val = train_test_split(
        X_combined, y_num, test_size=0.2, shuffle=False
    )
    _, _, y_col_train, y_col_val = train_test_split(
        X_combined, y_col, test_size=0.2, shuffle=False
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, "scaler.pkl")
    
    # ── Check how many unique values in each training target ──
    unique_bs_train = set(y_bs_train)
    unique_num_train = set(y_num_train)
    unique_col_train = set(y_col_train)
    
    print(f"\n[DEBUG] Training samples: {len(X_train)}")
    print(f"[DEBUG] BS classes: {unique_bs_train}, Num classes: {unique_num_train}, Col classes: {unique_col_train}")
    
    # ════════════════════════════════════════════
    # BIGSMALL MODELS (binary classification)
    # ════════════════════════════════════════════
    print("\n🧠 Training BIGSMALL models...")
    models_bs = {}
    val_scores_bs = {}
    
    for name, clf, kwargs in [
        ("RF", RandomForestClassifier, {"n_estimators": 200, "max_depth": 5, "random_state": 42}),
        ("ET", ExtraTreesClassifier, {"n_estimators": 200, "max_depth": 6, "random_state": 42}),
        ("BAG", BaggingClassifier, {"estimator": DecisionTreeClassifier(max_depth=5), "n_estimators": 100, "random_state": 42}),
        ("ADA", AdaBoostClassifier, {"n_estimators": 100, "learning_rate": 0.05, "random_state": 42}),
        ("DT", DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
        ("HGB", HistGradientBoostingClassifier, {"max_depth": 3, "learning_rate": 0.05, "random_state": 42}),
        ("KNN", KNeighborsClassifier, {"n_neighbors": min(7, len(unique_bs_train)), "weights": "distance", "metric": "manhattan"}),
        ("LR", LogisticRegression, {"C": 0.5, "solver": "liblinear"}),
    ]:
        model, acc = _train_single_classifier(clf, kwargs, X_train, y_bs_train, X_val, y_bs_val, name)
        if model is not None:
            models_bs[name] = model
            val_scores_bs[name] = acc
            #print(f"  ✅ {name} BS: {acc:.3f}")
    
    # Models that need scaling are wrapped so training and inference use the
    # exact same transformation.
    for name, clf, kwargs in [
        ("SVM", SVC, {"kernel": "rbf", "probability": True, "C": 2, "gamma": "scale", "random_state": 42}),
        ("RIDGE", RidgeClassifier, {"alpha": 1.0, "random_state": 42}),
        ("PA", PassiveAggressiveClassifier, {"max_iter": 500, "C": 0.5, "random_state": 42}),
    ]:
        try:
            model = make_pipeline(StandardScaler(), clf(**kwargs)).fit(X_train, y_bs_train)
            if hasattr(model, "predict_proba"):
                acc = model.score(X_val, y_bs_val)
            else:
                preds = model.predict(X_val)
                acc = np.mean(preds == y_bs_val)
            models_bs[name] = model
            val_scores_bs[name] = acc
           #print(f"  ✅ {name} BS: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ {name} BS failed: {e}")
    
    # CatBoost BS
    try:
        model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                                    loss_function='Logloss', verbose=False, random_seed=42)
        model.fit(X_train, y_bs_train, eval_set=(X_val, y_bs_val), verbose=False)
        acc = model.score(X_val, y_bs_val)
        models_bs["CatBoost"] = model
        val_scores_bs["CatBoost"] = acc
        #print(f"  ✅ CatBoost BS: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ CatBoost BS failed: {e}")
    
    # LGBM BS
    try:
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                num_leaves=15, verbose=-1, random_state=42)
        model.fit(X_train, y_bs_train, eval_set=[(X_val, y_bs_val)],
                  callbacks=[lgbm_early_stopping(10)])
        acc = model.score(X_val, y_bs_val)
        models_bs["LGBM"] = model
        val_scores_bs["LGBM"] = acc
        #print(f"  ✅ LGBM BS: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ LGBM BS failed: {e}")
    
    # GPC
    if len(unique_bs_train) >= 2:
        try:
            kernel = C(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessClassifier(kernel=kernel, random_state=42).fit(X_train, y_bs_train)
            acc = model.score(X_val, y_bs_val)
            models_bs["GPC"] = model
            val_scores_bs["GPC"] = acc
            #print(f"  ✅ GPC BS: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ GPC BS failed: {e}")
    
    # LabelPropagation
    try:
        model = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
        model.fit(X_train, y_bs_train)
        acc = model.score(X_val, y_bs_val)
        models_bs["LBLPROP"] = model
        val_scores_bs["LBLPROP"] = acc
        #print(f"  ✅ LBLPROP BS: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ LBLPROP BS failed: {e}")
    
    # HMM
    if len(bs_seq) > 20:
        try:
            hmm = GaussianHMM(n_components=2, n_iter=300, random_state=42)
            hmm.fit(np.array(bs_seq).reshape(-1, 1))
            models_bs["HMM"] = hmm
            print(f"  ✅ HMM BS trained")
        except Exception as e:
            print(f"  ⚠️ HMM BS failed: {e}")
     
    proba_models = [(n, m) for n, m in models_bs.items() if hasattr(m, "predict_proba")]
    if len(proba_models) >= 3:
        try:
            voter = VotingClassifier(estimators=proba_models[:10], voting='soft')
            voter.fit(X_train, y_bs_train)
            acc = voter.score(X_val, y_bs_val)
            models_bs["VOTER"] = voter
            val_scores_bs["VOTER"] = acc
            #print(f"  ✅ VOTER BS: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ VOTER BS failed: {e}")
    
    #print("\n🧠 Training NUMBER models...")
    models_num = {}
    val_scores_num = {}
    
    num_ints_train = [decode_num(x) for x in y_num_train]
    num_ints_val = [decode_num(x) for x in y_num_val]
    unique_num_vals = set(num_ints_train)
    
    if len(unique_num_vals) < 2:
        print(f"  ⏭️ NUMBER training skipped: only {len(unique_num_vals)} unique value(s) in training set")
    else:
        #print(f"  [NUM] Training with {len(unique_num_vals)} unique classes in {len(num_ints_train)} samples")
        
        for name, clf, kwargs in [
            ("RF", RandomForestClassifier, {"n_estimators": 200, "max_depth": 8, "random_state": 42}),
            ("ET", ExtraTreesClassifier, {"n_estimators": 200, "max_depth": 8, "random_state": 42}),
            ("KNN", KNeighborsClassifier, {"n_neighbors": min(7, len(unique_num_vals)), "weights": "distance"}),
            ("HGB", HistGradientBoostingClassifier, {"max_depth": 3, "learning_rate": 0.05, "random_state": 42}),
        ]:
            model, acc = _train_single_classifier(clf, kwargs, X_train, num_ints_train, X_val, num_ints_val, f"{name} NUM")
            if model is not None:
                models_num[name] = model
                val_scores_num[name] = acc
                #print(f"  ✅ {name} NUM: {acc:.3f}")
        
        # CatBoost NUM (multi-class)
        try:
            model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                                        loss_function='MultiClass', verbose=False, random_seed=42)
            model.fit(X_train, num_ints_train, eval_set=(X_val, num_ints_val), verbose=False)
            acc = model.score(X_val, num_ints_val)
            models_num["CatBoost"] = model
            val_scores_num["CatBoost"] = acc
            #print(f"  ✅ CatBoost NUM: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ CatBoost NUM failed: {e}")
        
        # LR NUM (multinomial) — needs at least 2 classes in training
        if len(unique_num_vals) >= 2:
            try:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial', max_iter=500),
                )
                model.fit(X_train, num_ints_train)
                acc = model.score(X_val, num_ints_val)
                models_num["LR"] = model
                val_scores_num["LR"] = acc
                #print(f"  ✅ LR NUM: {acc:.3f}")
            except Exception as e:
                print(f"  ⚠️ LR NUM failed: {e}")
    
    # ════════════════════════════════════════════
    # COLOR MODELS (multi-class, 4 classes: 3-6)
    # ════════════════════════════════════════════
    #print("\n🧠 Training COLOR models...")
    models_col = {}
    val_scores_col = {}
    
    col_train = list(y_col_train)
    col_val = list(y_col_val)
    unique_col_vals = set(col_train)
    
    if len(unique_col_vals) < 2:
        print(f"  ⏭️ COLOR training skipped: only {len(unique_col_vals)} unique value(s) in training set")
    else:
        #print(f"  [COL] Training with {len(unique_col_vals)} unique classes in {len(col_train)} samples")
        
        for name, clf, kwargs in [
            ("RF", RandomForestClassifier, {"n_estimators": 200, "max_depth": 6, "random_state": 42}),
            ("ET", ExtraTreesClassifier, {"n_estimators": 200, "max_depth": 6, "random_state": 42}),
            ("KNN", KNeighborsClassifier, {"n_neighbors": min(5, len(unique_col_vals)), "weights": "distance"}),
            ("HGB", HistGradientBoostingClassifier, {"max_depth": 3, "learning_rate": 0.05, "random_state": 42}),
        ]:
            model, acc = _train_single_classifier(clf, kwargs, X_train, col_train, X_val, col_val, f"{name} COL")
            if model is not None:
                models_col[name] = model
                val_scores_col[name] = acc
                #print(f"  ✅ {name} COL: {acc:.3f}")
        
        # CatBoost COL
        try:
            model = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                        loss_function='MultiClass', verbose=False, random_seed=42)
            model.fit(X_train, col_train, eval_set=(X_val, col_val), verbose=False)
            acc = model.score(X_val, col_val)
            models_col["CatBoost"] = model
            val_scores_col["CatBoost"] = acc
            #print(f"  ✅ CatBoost COL: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ CatBoost COL failed: {e}")
    
    # ── Summary ──
    #if val_scores_bs:
        #print("\n📊 BIGSMALL Validation:")
        #for name, acc in sorted(val_scores_bs.items(), key=lambda x: x[1], reverse=True):
            #print(f"  {short_names.get(name, name):10s}: {acc:.3f}")
    #if val_scores_num:
        #print("\n📊 NUMBER Validation:")
        #for name, acc in sorted(val_scores_num.items(), key=lambda x: x[1], reverse=True):
            #print(f"  {short_names.get(name, name):10s}: {acc:.3f}")
    #if val_scores_col:
        #print("\n📊 COLOR Validation:")
        #for name, acc in sorted(val_scores_col.items(), key=lambda x: x[1], reverse=True):
            #print(f"  {short_names.get(name, name):10s}: {acc:.3f}")
    
    # At the END of train_models(), change this:
    all_models = {'bs': models_bs, 'num': models_num}  # REMOVED 'col'
    joblib.dump(all_models, "models.pkl")
    #print(f"\n✅ Saved models: BS({len(models_bs)}), NUM({len(models_num)})")

def _filter_good_models(model_votes, model_stats_dict):
    """Filter models by accuracy threshold — but only after enough predictions."""
    global total_predictions
    
    # ── KEY FIX: Don't filter until we have enough predictions ──
    if total_predictions < FILTER_AFTER_PREDICTIONS:
        return list(model_votes.keys())
    
    good = []
    for name in model_votes:
        stats = model_stats_dict.get(name, {"recent_total": 0, "recent_correct": 0})
        if stats["recent_total"] > 0:
            acc = stats["recent_correct"] / stats["recent_total"]
            if acc >= MODEL_ACCURACY_THRESHOLD:
                good.append(name)
        else:
            good.append(name)
    
    if not good:
        good = list(model_votes.keys())
    return good


def _print_model_accs(model_votes, model_stats_dict, header=""):
    if header:
        print(header)
    log_line = ""
    for name in model_votes:
        stats = model_stats_dict.get(name, {"recent_total": 0, "recent_correct": 0})
        if stats["recent_total"] > 0:
            acc = stats["recent_correct"] / stats["recent_total"]
            text = f" {short_names.get(name, name)}: {acc:.0%}|"
        else:
            text = f" {short_names.get(name, name)}: No data|"
        print(text, end="")
        log_line += text

    with open("model_log.txt", "a") as f:
        f.write(log_line + "\n")

def _print_ai_agent_accs():
    """Read agent_ai_state.json and print each AI agent's individual performance
    with per-round correct/incorrect status, matching the ensemble style."""
    state_path = "agent_ai_state.json"
    if not os.path.exists(state_path):
        return
    try:
        import json
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agents = data.get("agents", {})
        if not agents:
            return
        print("\n📊 AI AGENT INDIVIDUAL PERFORMANCE:\n")
        for name, state in sorted(agents.items()):
            total = state.get("total", 0)
            correct = state.get("correct", 0)
            recent = state.get("recent", [])
            
            # The last element in recent tells us if the most recent prediction was correct
            last_correct = recent[-1] if recent else None
            if last_correct == 1:
                icon = "✅"
            elif last_correct == 0:
                icon = "❌"
            else:
                icon = "⏳"
            
            acc = (correct / total * 100) if total > 0 else 0.0
            
            result_text = "Correct" if last_correct == 1 else "Incorrect"
            print(f"  [AI] {name:25s} {icon} {result_text} | 📈 Accuracy: {acc:.1f}% ({correct}/{total})")
       
    except Exception as e:
        print(f"  [AI Agents] Could not read state: {e}")

def map_color_to_bigsmall_recent(predicted_color, df, lookback=30):
    color_keywords = {3: ["green"], 4: ["red"]}
    
    # Compound colors (5, 6) already carry BigSmall — handle directly
    if predicted_color == 5:  # red & purple6
        return 0, "small"
    elif predicted_color == 6:  # green & purple
        return 5, "big"
    
    keywords = color_keywords.get(predicted_color)
    if not keywords:
        return None, None
    
    # Get the most recent rows
    recent_df = df.tail(lookback)
    
    # ⭐ CRITICAL: iterate from the VERY END (most recent) backwards
    for idx in range(len(recent_df) - 1, -1, -1):
        row = recent_df.iloc[idx]
        row_color = str(row["Color"]).strip().lower()
        
        # Skip compound colors (violet/purple) — they're codes 5/6, not pure red/green
        if "violet" in row_color or "purple" in row_color:
            continue
        
        # Check if this row's color contains our keyword
        if any(kw in row_color for kw in keywords):
            target_number = int(row["Numbers"])
            target_bs = str(row["BigSmall"]).strip().lower()
            print(f"  → Found '{row_color}' at row index {recent_df.index[idx]} "
                  f"(Draw {row.get('DrawNo', '?')}): "
                  f"Number={target_number}, BigSmall={target_bs}")
            return target_number, target_bs
    
    # Fallback: if no pure color found in recent data
    print(f"  ⚠️ No '{keywords[0]}' found in last {lookback} rows, using defaults")
    if predicted_color == 4:  # red
        # Check recent BigSmall trend for smarter fallback
        recent_big_count = sum(1 for _, r in recent_df.iterrows()
                               if str(r.get("BigSmall", "")).strip().lower() == "big")
        recent_small_count = lookback - recent_big_count
        if recent_small_count > recent_big_count:
            return 2, "small"  # small-biased fallback
        return 8, "big"
    elif predicted_color == 3:  # green
        return 7, "big"  # default green big

def train_color_models_only(csv_file):
    df = pd.read_csv(csv_file)
    df = df[df["BigSmall"].isin(["big", "small"])].copy()
    
    if len(df) < 10:
        print("⚠️ Not enough data for color training")
        return {}, {}
    
    # Encode colors
    col_seq = df["Color"].apply(lambda x: encode_col(x) if isinstance(x, str) else 4).tolist()
    
    #print(f"  [COL] Unique colors in training data: {sorted(set(col_seq))}")
    
    if len(set(col_seq)) < 2:
        print(f"⏭️ Only {len(set(col_seq))} unique color(s) — skipping training")
        return {}, {}
    
    # Build features: window of past colors → next color
    X, y = [], []
    for i in range(MODEL_WINDOW, len(col_seq)):
        X.append(col_seq[i-MODEL_WINDOW:i])
        y.append(col_seq[i])
    
    if len(X) < 10:
        print("⚠️ Not enough color sequences")
        return {}, {}
    
    X = np.array(X)
    y = np.array(y)
    
    unique_colors = set(y)
    #print(f"\n🧠 Training COLOR models on {len(unique_colors)} unique colors, {len(X)} samples")
    #print(f"   Color classes: {sorted(unique_colors)}")
    
    if len(unique_colors) < 2:
        print("⏭️ Only 1 color class — skipping training")
        return {}, {}
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    models_col = {}
    val_scores_col = {}
    
    for name, clf, kwargs in [
        ("RF", RandomForestClassifier, {"n_estimators": 200, "max_depth": 6, "random_state": 42}),
        ("ET", ExtraTreesClassifier, {"n_estimators": 200, "max_depth": 6, "random_state": 42}),
        ("KNN", KNeighborsClassifier, {"n_neighbors": min(5, len(unique_colors)), "weights": "distance"}),
        ("HGB", HistGradientBoostingClassifier, {"max_depth": 3, "learning_rate": 0.05, "random_state": 42}),
        ("DT", DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
    ]:
        try:
            model = clf(**kwargs).fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            models_col[name] = model
            val_scores_col[name] = acc
            #print(f"  ✅ {name} COL: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ {name} COL failed: {e}")
    
    # CatBoost COL
    try:
        model = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                    loss_function='MultiClass', verbose=False, random_seed=42)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        acc = model.score(X_val, y_val)
        models_col["CatBoost"] = model
        val_scores_col["CatBoost"] = acc
        #print(f"  ✅ CatBoost COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ CatBoost COL failed: {e}")

    # ADA - AdaBoost
    try:
        model = AdaBoostClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["ADA"] = model
        val_scores_col["ADA"] = acc
        #print(f"  ✅ ADA COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ ADA COL failed: {e}")

    # BAG - Bagging
    try:
        model = BaggingClassifier(n_estimators=200, max_samples=0.8, max_features=0.8, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["BAG"] = model
        val_scores_col["BAG"] = acc
        #print(f"  ✅ BAG COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ BAG COL failed: {e}")

    try:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        model = GaussianProcessClassifier(kernel=kernel, random_state=42, max_iter_predict=100)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["GPC"] = model
        val_scores_col["GPC"] = acc
        #print(f"  ✅ GPC COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ GPC COL failed: {e}")
 
    try:
        from hmmlearn import hmm
        
        
        hmm_models = {}
        for color_val in unique_colors:
            color_seqs = [col_seq[i-MODEL_WINDOW:i] for i in range(MODEL_WINDOW, len(col_seq)) if col_seq[i] == color_val]
            if len(color_seqs) < 5:
                continue
            hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
            hmm_model.fit(np.array(color_seqs))
            hmm_models[color_val] = hmm_model
        
        if hmm_models:
            wrapper = HMMWrapper(hmm_models, list(unique_colors))
            y_pred = wrapper.predict(X_val)
            acc = np.mean(y_pred == y_val)
            models_col["HMM"] = wrapper
            val_scores_col["HMM"] = acc
            #print(f"  ✅ HMM COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ HMM COL failed: {e}")

    try:
        from sklearn.ensemble import VotingClassifier
        
        # Collect all trained estimators (skip HMM since it's non-standard)
        estimators = []
        for name in models_col:
            if name == "HMM":
                continue
            estimators.append((name, models_col[name]))
        
        if len(estimators) >= 3:
            voter = VotingClassifier(estimators=estimators, voting='soft' if hasattr(estimators[0][1], "predict_proba") else 'hard')
            voter.fit(X_train, y_train)
            acc = voter.score(X_val, y_val)
            models_col["VOTER"] = voter
            val_scores_col["VOTER"] = acc
            #print(f"  ✅ VOTER COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ VOTER COL failed: {e}")

    try:
        scaler_lr = StandardScaler()
        X_train_scaled = scaler_lr.fit_transform(X_train)
        X_val_scaled = scaler_lr.transform(X_val)
        model = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial', max_iter=500)
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_val_scaled, y_val)
        models_col["LR"] = model
        val_scores_col["LR"] = acc
        #print(f"  ✅ LR COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ LR COL failed: {e}")

    # PA - PassiveAggressiveClassifier
    try:
        from sklearn.linear_model import PassiveAggressiveClassifier
        model = PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=42, tol=1e-3)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["PA"] = model
        val_scores_col["PA"] = acc
        #print(f"  ✅ PA COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ PA COL failed: {e}")

   
    try:
        model = LabelPropagation(kernel='knn', n_neighbors=min(5, len(unique_colors)), max_iter=300)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["LBLPROP"] = model
        val_scores_col["LBLPROP"] = acc
        #print(f"  ✅ LBLPROP COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ LBLPROP COL failed: {e}")

    try:
        model = GaussianNB()
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["NGRAM"] = model
        val_scores_col["NGRAM"] = acc
        #print(f"  ✅ NGRAM COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ NGRAM COL failed: {e}")

    try:
        scaler_svm = StandardScaler()
        X_train_scaled = scaler_svm.fit_transform(X_train)
        X_val_scaled = scaler_svm.transform(X_val)
        model = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovo', random_state=42)
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_val_scaled, y_val)
        models_col["SVM"] = model
        val_scores_col["SVM"] = acc
        #print(f"  ✅ SVM COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ SVM COL failed: {e}")

    try:
        model = RidgeClassifier(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["RIDGE"] = model
        val_scores_col["RIDGE"] = acc
        #print(f"  ✅ RIDGE COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ RIDGE COL failed: {e}")

    try:
        model = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               num_leaves=31, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_col["LGBM"] = model
        val_scores_col["LGBM"] = acc
        #print(f"  ✅ LGBM COL: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ LGBM COL failed: {e}")
    
    if val_scores_col:
        #print("\n📊 COLOR Validation:")
        for name, acc in sorted(val_scores_col.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:10s}: {acc:.3f}")
    
    return models_col, val_scores_col
    
def predict_all_outputs(history_bs, history_num, history_col, models_bs, models_num, models_col):
    
    full_bs = [encode_bs(x) if isinstance(x, str) else x for x in history_bs]
    full_num = [decode_num(x) if isinstance(x, str) else x for x in history_num]
    full_col = list(history_col)  # already encoded 3,4,5,6
    
    if len(full_bs) < MODEL_WINDOW:
        return "WAIT", {}, 0.0, None
    
    # Ensure all same length
    min_len = min(len(full_bs), len(full_num), len(full_col))
    full_bs = full_bs[-min_len:]
    full_num = full_num[-min_len:]
    full_col = full_col[-min_len:]
    
    # Feature vectors
    seq_bs = full_bs[-MODEL_WINDOW:]
    seq_num = full_num[-MODEL_WINDOW:]
    seq_col = full_col[-MODEL_WINDOW:]
    
    # Combined feature (36-dim): BS(12) + Num(12) + Col(12) → for BS models
    feature_vector_combined = list(seq_bs) + list(seq_num) + list(seq_col)
    X_combined = np.array(feature_vector_combined).reshape(1, -1)
    
    # Number-only feature (12-dim): just numbers → for num models
    X_num = np.array(seq_num).reshape(1, -1)
    
    # Color-only feature (12-dim): just colors → for col models
    X_col = np.array(seq_col).reshape(1, -1)
    
    # Collect ALL model names across all 3 dicts
    all_model_names = set()
    all_model_names.update(models_bs.keys())
    all_model_names.update(models_num.keys())
    all_model_names.update(models_col.keys())
    
    # Also add heuristic models
    all_model_names.add("Markov")
    all_model_names.add("NGRAM")
    all_model_names.add("RegimeTrend")
    
    # ── STEP 1 diagnostic: number-model class coverage (warning only) ──
    # If any saved number model no longer covers the full 0-9 range, flag it
    # now so the value/range health is visible instead of silently mispredicting.
    try:
        _incomplete = []
        for _nm, _ in sorted(models_num.items()):
            _cls = _model_classes(_)
            if _cls is not None:
                _digits = sorted(int(c) for c in _cls)
                if _digits != list(range(10)):
                    _incomplete.append(f"{_nm}({_digits})")
        if _incomplete:
            print(f"  [STEP1] ⚠️ Number models missing full 0-9 coverage: {', '.join(_incomplete)}")
    except Exception as _e:
        print(f"  [STEP1] ⚠️ Coverage check failed: {_e}")

    # LEVEL 1: Per-model internal voting
    per_model_final = {}   # model_name → internal_consensus_bs
    per_model_details = {} # model_name → {bs, num, col, num_bs, col_bs}
    
 
    for name in sorted(all_model_names):
        # Get BS prediction
        bs_pred = None
        if name in models_bs:
            m = models_bs[name]
            try:
                if name == "HMM":
                    hmm_seq = np.array(full_bs).reshape(-1, 1)
                    hidden = m.predict(hmm_seq)
                    bs_pred = int(hidden[-1])
                else:
                    if hasattr(m, "predict_proba"):
                        prob = m.predict_proba(X_combined)[0]
                        classes = _model_classes(m)
                        if classes is not None:
                            bs_pred = int(classes[int(np.argmax(prob))])
                    else:
                        pred = m.predict(X_combined)
                        if hasattr(pred, '__iter__'):
                            bs_pred = int(pred[0])
                        else:
                            bs_pred = int(pred)
                        if bs_pred == -1:
                            bs_pred = 0
            except Exception as e:
                print(f"  ⚠️ {name} BS failed: {e}")
        
        # Get Number prediction
        num_pred = None
        if name in models_num:
            m = models_num[name]
            try:
                if hasattr(m, "predict_proba"):
                    prob = m.predict_proba(X_num)[0]
                    classes = _model_classes(m)
                    if classes is not None:
                        num_pred = int(classes[int(np.argmax(prob))])
                    else:
                        num_pred = int(np.argmax(prob))
                    # STEP 1: enforce valid 0-9 range. If classes_ is ever
                    # missing or sparse during a retrain, argmax/classes alone
                    # could emit a wrong or out-of-range "number"; the clamp
                    # guarantees the value/range stays good for prediction.
                    num_pred = int(min(9, max(0, num_pred)))
                else:
                    pred = m.predict(X_num)
                    if hasattr(pred, '__iter__'):
                        num_pred = int(pred[0])
                    else:
                        num_pred = int(pred)
            except Exception as e:
                print(f"  ⚠️ {name} NUM failed: {e}")
        
        # Get Color prediction
        col_pred = None
        if name in models_col:
            m = models_col[name]
            try:
                if hasattr(m, "predict_proba"):
                    prob = m.predict_proba(X_col)[0]
                    pred_idx = int(np.argmax(prob))
                    classes = _model_classes(m)
                    if classes is not None:
                        col_pred = int(classes[pred_idx])
                    else:
                        col_pred = pred_idx
                else:
                    pred = m.predict(X_col)
                    if hasattr(pred, '__iter__'):
                        col_pred = int(pred[0])
                    else:
                        col_pred = int(pred)
            except Exception as e:
                print(f"  ⚠️ {name} COL failed: {e}")
        
        # Heuristic models
        # Fallback helper: most-frequent digit in the recent window so every
        # heuristic model always produces a number vote (per the full-pass goal).
        def _seen_fallback():
            return Counter(full_num[-MODEL_WINDOW:] or full_num).most_common(1)[0][0] if full_num else None

        if name == "Markov":
            bs_pred = markov_predict(full_bs)
            if len(full_num) >= 2:
                transitions = {}
                for i in range(len(full_num) - 1):
                    key = (full_num[i], full_num[i+1])
                    transitions[key] = transitions.get(key, 0) + 1
                last_n = full_num[-1]
                cand = [(n, c) for (l, n), c in transitions.items() if l == last_n]
                if cand:
                    num_pred = max(cand, key=lambda x: x[1])[0]
                else:
                    num_pred = _seen_fallback()   # no transition from last digit
            else:
                num_pred = _seen_fallback()
            if len(full_col) >= 2:
                transitions = {}
                for i in range(len(full_col) - 1):
                    key = (full_col[i], full_col[i+1])
                    transitions[key] = transitions.get(key, 0) + 1
                last_c = full_col[-1]
                cand = [(n, c) for (l, n), c in transitions.items() if l == last_c]
                if cand:
                    col_pred = max(cand, key=lambda x: x[1])[0]
        
        if name == "NGRAM":
            bs_pred = ngram_predict(full_bs, n=3)
            ng_num, _ = ngram_predict_num(full_num, n=3)
            if ng_num is not None:
                num_pred = ng_num
            elif name in models_num:
                # Fall back to the trained GaussianNB number model when the
                # n-gram has no recorded match (so NGRAM always votes).
                m = models_num[name]
                try:
                    if hasattr(m, "predict_proba"):
                        prob = m.predict_proba(X_num)[0]
                        cls = _model_classes(m)
                        num_pred = int(cls[int(np.argmax(prob))]) if cls is not None else int(np.argmax(prob))
                    else:
                        pred = m.predict(X_num)
                        num_pred = int(pred[0]) if hasattr(pred, '__iter__') else int(pred)
                except Exception:
                    num_pred = _seen_fallback()
            else:
                num_pred = _seen_fallback()
            col_pred = ngram_predict_col(full_col, n=3)
        
        if name == "RegimeTrend":
            bs_pred = predict_regime_trend(full_bs)
            if len(full_num) >= 20:
                num_pred = Counter(full_num[-20:]).most_common(1)[0][0]
            else:
                num_pred = _seen_fallback()
            if len(full_col) >= 20:
                col_pred = Counter(full_col[-20:]).most_common(1)[0][0]
        
        # ⭐ KEY FIX: Derive BS from Number using simple rule
        num_bs = None
        # HMM is a GaussianHMM state model (no digit classifier). Give it a
        # most-frequent-recent-digit number vote so it also joins the number
        # ensemble instead of showing Num:None.
        if name == "HMM" and num_pred is None:
            num_pred = Counter(full_num[-MODEL_WINDOW:] or full_num).most_common(1)[0][0] if full_num else None
        if num_pred is not None:
            num_bs = number_to_bigsmall(num_pred)  # 0-4 → 0(small), 5-9 → 1(big)
        
        # Color alone does not determine BIG/SMALL for pure red/green, so it
        # must never become an arbitrary BIG/SMALL vote. Compound colors are
        # retained for display only; the number channel is deterministic.
        col_bs = None

        # LEVEL 1: Internal vote uses only direct BS and deterministic Num→BS.
        internal_votes = [v for v in [bs_pred, num_bs] if v is not None]
        
        if internal_votes:
            internal_ones = sum(internal_votes)
            internal_zeros = len(internal_votes) - internal_ones
            internal_consensus = 1 if internal_ones > internal_zeros else 0
            internal_conf = max(internal_ones, internal_zeros) / len(internal_votes)
        else:
            internal_consensus = None
            internal_conf = 0.0
        
        per_model_final[name] = internal_consensus
        per_model_details[name] = {
            'bs': bs_pred, 'num': num_pred, 'col': col_pred,
            'num_bs': num_bs, 'col_bs': col_bs,
            'internal_consensus': internal_consensus,
            'internal_conf': internal_conf
        }
        
        # Print this model's internal voting
        bs_str      = f"{decode_bs(bs_pred):5s}" if bs_pred is not None else "None "
        num_str     = f"{num_pred}" if num_pred is not None else "None"
        col_str     = f"{COLOR_DECODE.get(col_pred, '?'):15s}" if col_pred is not None else "None          "
        num_bs_str  = f"{decode_bs(num_bs):5s}" if num_bs is not None else "None "
        col_bs_str  = f"{decode_bs(col_bs):5s}" if col_bs is not None else "None "
        
        vote_str = f"({internal_ones}/{len(internal_votes)})" if internal_votes else "(0/0)"

    
    print(f"\n{'='*65}")
    print(f"  LEVEL 2 — EXTERNAL ENSEMBLE VOTING")
    print(f"{'='*65}")
    
    # Filter only models that had valid internal consensus
    external_votes_raw = [v for v in per_model_final.values() if v is not None]
    
    if not external_votes_raw:
        print("  ⚠️ No model produced a valid internal consensus")
        return "WAIT", {}, 0.0, None
    
    # Apply accuracy filtering if past warmup
    global total_predictions
    if total_predictions >= FILTER_AFTER_PREDICTIONS:
        filtered_votes = []
        for name, consensus in per_model_final.items():
            if consensus is None:
                continue
            stats = model_stats_bs.get(name, {"recent_total": 0, "recent_correct": 0})
            if stats["recent_total"] > 0:
                acc = stats["recent_correct"] / stats["recent_total"]
                if acc >= MODEL_ACCURACY_THRESHOLD:
                    filtered_votes.append(consensus)
                else:
                    print(f"  ⏭️ {name:10s} filtered out (acc:{acc:.0%} < {MODEL_ACCURACY_THRESHOLD:.0%})")
            else:
                filtered_votes.append(consensus)
        
        external_votes = filtered_votes if filtered_votes else external_votes_raw
    else:
        external_votes = external_votes_raw
    
    # Count votes
    ones = sum(external_votes)
    zeros = len(external_votes) - ones
    final_pred = 1 if ones > zeros else 0
    final_conf = max(ones, zeros) / len(external_votes) if external_votes else 0.0
    
    # Define your favorite/special models
    SPECIAL_MODELS = {"RegimeTrend", "RIDGE", "DT"}

    print(f"\n  BIG votes ({ones}):")
    for name, consensus in sorted(per_model_final.items()):
        if consensus == 1 and name not in SPECIAL_MODELS:
            d = per_model_details[name]
            num_bs_display = decode_bs(d['num_bs']) if d['num_bs'] is not None else "NONE"
            print(f"    {name:10s}  Num:{d['num']}  →  {num_bs_display:5s} ")

    if ones == 0:
        print("    (none)")

    print(f"\n  SMALL votes ({zeros}):")
    for name, consensus in sorted(per_model_final.items()):
        if consensus == 0 and name not in SPECIAL_MODELS:
            d = per_model_details[name]
            num_bs_display = decode_bs(d['num_bs']) if d['num_bs'] is not None else "NONE"
            print(f"    {name:10s}  Num:{d['num']}  →  {num_bs_display:5s} ")

    if zeros == 0:
        print("    (none)")

 
    print(f"  ⭐ SPECIAL MODELS")
 
    for name in sorted(SPECIAL_MODELS):
        if name not in per_model_details:
            continue
        d = per_model_details[name]
        consensus = per_model_final.get(name)
        if consensus is None:
            vote_label = "ABSTAIN"
        else:
            vote_label = decode_bs(consensus)
        num_bs_display = decode_bs(d['num_bs']) if d['num_bs'] is not None else "NONE"
        print(f"    {name:15s}  Num:{d['num']}  →  {num_bs_display:5s}  ")

    # ════════════════════════════════════════════
    # NUMBER ENSEMBLE PREDICTION
    # ════════════════════════════════════════════
    num_votes = []
    for name, d in per_model_details.items():
        if d.get('num') is not None:
            num_votes.append(d['num'])
    
    if num_votes:
        num_counter = Counter(num_votes)
        final_num_pred = num_counter.most_common(1)[0][0]
        # STEP 1: defensive clamp on the final digit. A single bad model vote
        # must never push the ensemble number out of the 0-9 range or feed an
        # invalid value into number_to_bigsmall()/Num→BS.
        final_num_pred = int(min(9, max(0, final_num_pred)))
        num_conf = num_counter.most_common(1)[0][1] / len(num_votes)
        num_bs_from_num = number_to_bigsmall(final_num_pred)
        print(f"\n  🔢 NUMBER ENSEMBLE: {final_num_pred} (votes: {len(num_votes)}, conf:{num_conf:.1%})")
        print(f"     Num→BS: {decode_bs(num_bs_from_num)}")
        # Show per-model number votes
        for digit, count in sorted(num_counter.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"     {digit}: {count} {bar}")
    # ── THE FINAL DECISION IS NOW DRIVEN BY THE NUMBER ──
        # Only the final number prediction matters; the Big/Small of that number
        # is THE final prediction. The Big/Small vote tally is not the decider.
        final_pred = num_bs_from_num
        final_conf = num_conf
    else:
        final_num_pred = None
        final_pred = None
        final_conf = 0.0
        print(f"\n  🔢 NUMBER ENSEMBLE: No predictions available")

    print(f"\n{'='*65}")
    if final_num_pred is not None:
        print(f"  🏆 FINAL DECISION (from NUMBER): {decode_bs(final_pred)}")
        print(f"    FINAL NUMBER: {final_num_pred}  →  {final_num_pred} is {decode_bs(final_pred)} (conf:{final_conf:.1%})")
    else:
        print("  🏆 FINAL DECISION: NONE (no number prediction this round)")
    print(f"{'='*65}")
    
    return final_pred, per_model_details, final_conf, final_num_pred

def update_stats(model_votes_dict, true_label, model_stats_dict):
    for name, pred in model_votes_dict.items():
        stats = model_stats_dict[name]
        stats["total"] += 1
        correct = int(pred == true_label)
        stats["correct"] += correct
        stats["recent_results"].append(correct)
        RECENT_WINDOW = 25
        if len(stats["recent_results"]) > RECENT_WINDOW:
            stats["recent_results"].pop(0)
        stats["recent_total"] = len(stats["recent_results"])
        stats["recent_correct"] = sum(stats["recent_results"])

def load_history(n=GLOBAL_DATA_WINDOW):
    if not os.path.exists(CSV_PATH):
        print("  [DEBUG] CSV file not found")
        return [], [], []
    
    df = pd.read_csv(CSV_PATH)
    
    if len(df) < 7:
        print(f"  [DEBUG] Only {len(df)} rows, need 7")
        return [], [], []
    
    # ── FORCE Numbers to clean integers ──
    df["Numbers"] = df["Numbers"].astype(str).str.strip()
    # Remove .0 suffix
    df["Numbers"] = df["Numbers"].str.replace(r"\.0$", "", regex=True)
    # Convert to int, NaN → 0
    df["Numbers"] = pd.to_numeric(df["Numbers"], errors='coerce').fillna(0).astype(int)
    # Convert back to string for consistency with rest of code
    df["Numbers_str"] = df["Numbers"].astype(str)
    

    recent = df.tail(n)
    
    bs_list = []
    nums_list = []
    colors_list = []
    skipped = 0
    
    for idx, row in recent.iterrows():
        bs_val = str(row.get("BigSmall", "")).strip().lower()
        num_val = row["Numbers"]  # Already int
        col_val = str(row.get("Color", "")).strip().lower()
        
        if bs_val in ("big", "small"):
            bs_list.append(bs_val)
            nums_list.append(num_val)  # Already int, no need for encode_num
            
            color_map = {"red": 4, "green": 3, "red & purple": 5, 
                         "green & purple": 6, "purple": 6}
            colors_list.append(color_map.get(col_val, 4))
        else:
            skipped += 1
    
    return bs_list, nums_list, colors_list

def load_existing_periods_once():
    periods = set()
    if not os.path.exists(CSV_PATH):
        return periods
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                periods.add(row[0])
    print(f"📦 Cached {len(periods)} existing periods")
    return periods

def check_retrain_needed():
    global rounds_since_retrain, MODELS
    rounds_since_retrain += 1
    if rounds_since_retrain >= RETRAIN_INTERVAL:
        print(f"\n🔄 Auto-retraining ALL models after {rounds_since_retrain} rounds...")
        try:
            # Retrain NUMBER-ONLY models (BS/Color removed — not used for the number decision)
            print("🧠 Retraining NUMBER-ONLY models...")
            number_only_models, _ = train_number_models_only(CSV_PATH)
            if number_only_models:
                joblib.dump(number_only_models, "number_models.pkl")
                MODELS['num'] = number_only_models
                print(f"✅ Number models retrained: {len(number_only_models)}")

            print("✅ Number models retrained successfully")
            rounds_since_retrain = 0
        except Exception as e:
            print(f"⚠️ Retrain failed: {e}")

def check_accuracy_with_latest_result():
    global total_predictions, correct_predictions, last_prediction_bs
    global last_model_predictions_bs, model_stats_bs
    global last_model_predictions_num, model_stats_num
    global last_model_predictions_col, model_stats_col
    global ai_total_predictions, ai_correct_predictions, last_ai_prediction_bs

    if last_prediction_bs is None:
        return

    history_bs, _, _ = load_history()
    if not history_bs:
        return

    actual = history_bs[-1]
    total_predictions += 1

    # ── ML Model Ensemble Accuracy ──
    pred_str = decode_bs(last_prediction_bs)
    if pred_str.lower() == actual.lower():
        correct_predictions += 1
        result_icon = "✅ Correct"
    else:
        result_icon = "❌ Incorrect"

    accuracy = (correct_predictions / total_predictions) * 100

    actual_encoded = encode_bs(actual)
    for name, pred in last_model_predictions_bs.items():
        update_stats({name: pred}, actual_encoded, model_stats_bs)

    if last_model_predictions_num:
        try:
            df = pd.read_csv(CSV_PATH)
            last_num_actual = int(df["Numbers"].iloc[-1])
            for name, pred in last_model_predictions_num.items():
                update_stats({name: pred}, last_num_actual, model_stats_num)
        except:
            pass

    if last_model_predictions_col:
        try:
            df = pd.read_csv(CSV_PATH)
            last_col_str = df["Color"].iloc[-1]
            last_col_actual = encode_col(last_col_str)
            for name, pred in last_model_predictions_col.items():
                update_stats({name: pred}, last_col_actual, model_stats_col)
        except:
            pass

    print(f"{result_icon} | 📈 Accuracy: {accuracy:.1f}% "
          f"({correct_predictions}/{total_predictions})")

    # ── Combined AI System Accuracy ──
    if last_ai_prediction_bs is not None:
        ai_total_predictions += 1
        ai_pred_str = decode_bs(last_ai_prediction_bs)
        if ai_pred_str.lower() == actual.lower():
            ai_correct_predictions += 1
            ai_result_icon = "✅"
        else:
            ai_result_icon = "❌"
        ai_accuracy = (ai_correct_predictions / ai_total_predictions) * 100
        print(f"  {ai_result_icon} | AI System Accuracy: {ai_accuracy:.1f}% "
              f"({ai_correct_predictions}/{ai_total_predictions})")


def train_number_models_only(csv_file):

    df = pd.read_csv(csv_file)
    df = df[df["BigSmall"].isin(["big", "small"])].copy()
    
    if len(df) < 10:
        print("⚠️ Not enough data for number training")
        return {}, {}
    
    # Clean numbers properly
    df["Numbers"] = df["Numbers"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["Numbers"] = pd.to_numeric(df["Numbers"], errors='coerce').fillna(0).astype(int)
    
    num_seq = df["Numbers"].tolist()
    
    if len(set(num_seq)) < 2:
        print(f"⏭️ Only {len(set(num_seq))} unique number(s) — skipping training")
        return {}, {}
    
    # Build features: window of past numbers → next number
    X, y = [], []
    for i in range(MODEL_WINDOW, len(num_seq)):
        X.append(num_seq[i-MODEL_WINDOW:i])  # Past 12 numbers
        y.append(num_seq[i])                  # Next number
    
    if len(X) < 10:
        print("⚠️ Not enough number sequences")
        return {}, {}
    
    X = np.array(X)
    y = np.array(y)
    
    unique_nums = set(y)
    print(f"\n🧠 Training NUMBER models on {len(unique_nums)} unique digits, {len(X)} samples")
    print(f"   Number classes: {sorted(unique_nums)}")
    
    if len(unique_nums) < 2:
        print("⏭️ Only 1 number class — skipping training")
        return {}, {}
    
    # Split (small data: use test_size=0.2 or lower)
    test_size = min(0.2, 5.0 / len(X))  # At least 5 validation samples
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    models_num = {}
    val_scores_num = {}
     
    try:
        heuristic_model = HeuristicPredictor(window=MODEL_WINDOW, num_seq=num_seq)
        y_pred = heuristic_model.predict(X_val)
        acc = np.mean(y_pred == y_val)
        models_num["HEUR"] = heuristic_model
        val_scores_num["HEUR"] = acc
        print(f"  ✅ HEUR NUM: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ HEUR failed: {e}")
     
    for name, clf, kwargs in [
        ("DT",    DecisionTreeClassifier,     {"max_depth": 4, "min_samples_leaf": 2, "random_state": 42}),
        ("RF",    RandomForestClassifier,     {"n_estimators": 50, "max_depth": 4, "min_samples_leaf": 2, "random_state": 42}),
        ("ET",    ExtraTreesClassifier,       {"n_estimators": 50, "max_depth": 4, "min_samples_leaf": 2, "random_state": 42}),
        ("KNN",   KNeighborsClassifier,       {"n_neighbors": min(3, len(unique_nums)), "weights": "uniform", "metric": "manhattan"}),
        ("HGB",   HistGradientBoostingClassifier, {"max_depth": 3, "learning_rate": 0.1, "min_samples_leaf": 5, "random_state": 42}),
        ("RIDGE", RidgeClassifier,            {"alpha": 10.0, "random_state": 42}),
        ("NGRAM", GaussianNB,                 {}),
        ("NC",    NearestCentroid,            {"metric": "euclidean"}),
        # ── Full-pass additions: give every present model a number classifier ──
        ("BAG",   BaggingClassifier,          {"estimator": DecisionTreeClassifier(max_depth=4, random_state=42), "n_estimators": 50, "random_state": 42}),
        ("ADA",   AdaBoostClassifier,         {"n_estimators": 50, "learning_rate": 0.1, "random_state": 42}),
        ("PA",    PassiveAggressiveClassifier,{"C": 1.0, "max_iter": 1000, "random_state": 42}),
        ("GPC",   GradientBoostingClassifier, {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 60, "random_state": 42}),
        ("LBLPROP", LabelPropagation,         {"kernel": "rbf", "gamma": 20, "max_iter": 1000}),
    ]:
        try:
            model = clf(**kwargs).fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            models_num[name] = model
            val_scores_num[name] = acc
            print(f"  ✅ {name} NUM: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ {name} NUM failed: {e}")
     
    if CATBOOST_AVAILABLE:
        try:
            model = CatBoostClassifier(
                iterations=100, depth=3, learning_rate=0.1,
                loss_function='MultiClass', verbose=False, random_seed=42
            )
            model.fit(X_train, y_train, verbose=False)
            acc = model.score(X_val, y_val)
            models_num["CatBoost"] = model
            val_scores_num["CatBoost"] = acc
            print(f"  ✅ CatBoost NUM: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ CatBoost NUM failed: {e}")
     
    if LGBM_AVAILABLE:
        try:
            model = LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                num_leaves=16, min_child_samples=3, random_state=42, verbose=-1
            )
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            models_num["LGBM"] = model
            val_scores_num["LGBM"] = acc
            print(f"  ✅ LGBM NUM: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ LGBM NUM failed: {e}")
     
    try:
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.01, solver='lbfgs', multi_class='multinomial',
                max_iter=1000, penalty='l2', random_state=42
            )
        )
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_val, y_val)
        models_num["LR"] = pipeline
        val_scores_num["LR"] = acc
        print(f"  ✅ LR NUM: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ LR NUM failed: {e}")
     
    try:
        pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', C=1.0, random_state=42)
        )
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_val, y_val)
        models_num["SVM"] = pipeline
        val_scores_num["SVM"] = acc
        print(f"  ✅ SVM NUM: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ SVM NUM failed: {e}")

    # ── Full-pass: add new strong number classifiers ──
    # MLP (neural net) — strong classifier with soft probability for voting
    try:
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16), activation='relu',
            alpha=0.01, max_iter=1000, random_state=42
        )
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_num["MLP"] = model
        val_scores_num["MLP"] = acc
        print(f"  ✅ MLP NUM: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ MLP NUM failed: {e}")

    # XGBoost — strong gradient boosting (if installed)
    if XGB_AVAILABLE:
        try:
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                reg_lambda=1.0, reg_alpha=0.1, random_state=42,
                eval_metric="mlogloss", use_label_encoder=False
            )
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            models_num["XGB"] = model
            val_scores_num["XGB"] = acc
            print(f"  ✅ XGB NUM: {acc:.3f}")
        except Exception as e:
            print(f"  ⚠️ XGB NUM failed: {e}")

    # GradientBoosting (an independent strong tree ensemble)
    try:
        model = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_samples_leaf=3, random_state=42
        )
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        models_num["GB"] = model
        val_scores_num["GB"] = acc
        print(f"  ✅ GB NUM: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ GB NUM failed: {e}")

    # VOTER — soft-vote ensemble of the number estimators (only those with soft prob)
    try:
        proba_names = [
            n for n in ["RF", "ET", "KNN", "HGB", "MLP"]
            if n in models_num and hasattr(models_num[n], "predict_proba")
        ]
        if len(proba_names) >= 3:
            est = [(n, models_num[n]) for n in proba_names]
            voter = VotingClassifier(estimators=est, voting="soft")
            voter.fit(X_train, y_train)
            acc = voter.score(X_val, y_val)
            models_num["VOTER"] = voter
            val_scores_num["VOTER"] = acc
            print(f"  ✅ VOTER NUM: {acc:.3f}")
    except Exception as e:
        print(f"  ⚠️ VOTER NUM failed: {e}")

    if val_scores_num:
        print("\n📊 NUMBER Validation Scores (sorted):")
        for name, acc in sorted(val_scores_num.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:10s}: {acc:.3f}")

    return models_num, val_scores_num

class HeuristicPredictor:
    """
    Progressive window matching heuristic.
    1) Try exact match with full window
    2) Shrink window progressively (12→11→10...→1)
    3) Fallback: return most frequent number in full sequence
    """
    def __init__(self, window=MODEL_WINDOW, num_seq=None):
        self.window = window
        self.num_seq = num_seq if num_seq is not None else []
    
    def fit(self, X, y):
        return self  # No training needed
    
    def predict(self, X):
        predictions = []
        for window in X:
            pred = self._find_next(tuple(map(int, window)))
            predictions.append(pred)
        return np.array(predictions)
    
    def _find_next(self, window_tuple):
        N = len(self.num_seq)
        w = len(window_tuple)
        
        # 1) Exact match with full window
        # Search from N-w-1 down to 0 (skip the query position itself)
        for i in range(N - w - 1, -1, -1):
            if tuple(self.num_seq[i:i+w]) == window_tuple:
                return self.num_seq[i + w]
        
        # 2) Progressive window shrinking (w-1, w-2, ..., 1)
        for shrink in range(1, min(w, 6)):  # Don't shrink below w-5
            sub = window_tuple[shrink:]
            sw = len(sub)
            for i in range(N - sw - 1, -1, -1):
                if tuple(self.num_seq[i:i+sw]) == sub:
                    return self.num_seq[i + sw]
        
        # 3) Last resort: most frequent number
        return Counter(self.num_seq).most_common(1)[0][0]
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


def main():
    global banner_printed, existing_periods, MODELS, HGB_WINDOW_MODEL, scaler
    global total_predictions, correct_predictions, last_prediction_bs
    global last_prediction_num, last_prediction_col
    global last_model_predictions_bs, last_model_predictions_num, last_model_predictions_col

    # Reset AI agent state so every session starts fresh (matches ensemble counters)
    state_path = "agent_ai_state.json"
    if os.path.exists(state_path):
        try:
            os.remove(state_path)
            print("[MULTI-AI] Previous agent state cleared for fresh session")
        except Exception as e:
            print(f"[MULTI-AI] Could not clear state: {e}")

    multi_agent = None
    if MultiAgentSystem is not None:
        try:
            multi_agent = MultiAgentSystem(CSV_PATH)
        except Exception as exc:
            print(f"[MULTI-AI] DISABLED - initialization failed: {exc}")

    if not banner_printed:
        banner_printed = True

    ensure_csv()
    bootstrap_capture_all_rows()
    existing_periods = load_existing_periods_once()

    # ── Train NUMBER-ONLY models (only these are used for prediction) ──
    number_only_models, _ = train_number_models_only(CSV_PATH)
    if number_only_models:
        joblib.dump(number_only_models, "number_models.pkl")

    # ── Load number models only (BS/Color removed — not used for the number decision) ──
    MODELS['num'] = {}
    if os.path.exists("number_models.pkl") and os.path.getsize("number_models.pkl") > 0:
        MODELS['num'] = joblib.load("number_models.pkl")
        print(f"✅ Number models loaded: {len(MODELS['num'])}")
    else:
        print("⚠️ number_models.pkl missing/empty")

    history_bs, history_num, history_col = load_history()
    seq_bs = [encode_bs(x) for x in history_bs]
    HGB_WINDOW_MODEL = train_hgb_window(seq_bs)

    try:
        if os.path.exists("scaler.pkl") and os.path.getsize("scaler.pkl") > 0:
            scaler = joblib.load("scaler.pkl")
        else:
            scaler = None
    except:
        scaler = None

    capture_post_bootstrap_rows()
    print("✅ Post-bootstrap rows captured. System fully synced.\n")
    
    try:
        while True:
            wait_for_capture_time()
            period, numbers, bigsmall, color = capture_row()
            if period:
                logged = log_row(period, numbers, bigsmall, color)
                if logged:
                    global BOOTSTRAP_MODE
                    if BOOTSTRAP_MODE:
                        print("🟡 Startup sync complete — prediction skipped once.")
                        BOOTSTRAP_MODE = False
                        continue

                    if last_prediction_bs is not None:
                        check_accuracy_with_latest_result()

                    # Score the previous multi-agent discussion against this
                    # newly logged real row before asking for the next one.
                    if multi_agent is not None:
                        try:
                            multi_agent.score_actual(
                                bigsmall, numbers, color, period=period
                            )
                        except Exception as exc:
                            pass

                    history_bs, history_num, history_col = load_history()

                    # ── CHANGED: Single unified prediction with 2-level voting ──
                    prediction_result, per_model_details, overall_conf, final_num_pred = predict_all_outputs(
                        history_bs, history_num, history_col,
                        MODELS.get('bs', {}),
                        MODELS.get('num', {}),
                        MODELS.get('col', {})
                    )

                    # The AI layer is deliberately after the existing model
                    # ensemble. Its headline output is Big/Small; number and
                    # color are supporting evidence only.
                    if multi_agent is not None:
                        try:
                            ai_result = multi_agent.analyze(
                                history_bs,
                                history_num,
                                history_col,
                                per_model_details,
                                overall_conf,
                            )
                            # Capture the combined AI system prediction for accuracy tracking
                            global last_ai_prediction_bs
                            ai_pred = ai_result.get("prediction") if ai_result else None
                            if ai_pred in ("BIG", "SMALL"):
                                last_ai_prediction_bs = 1 if ai_pred == "BIG" else 0
                            else:
                                last_ai_prediction_bs = None
                        except Exception as exc:
                            print(f"[MULTI-AI] Analysis failed; existing prediction retained: {exc}")
                            last_ai_prediction_bs = None
                    else:
                        last_ai_prediction_bs = None

                    # ── NEW: Print individual AI agent performance ──
                    _print_ai_agent_accs()

                    if isinstance(prediction_result, int) and overall_conf >= CONFIDENCE_THRESHOLD:
                        last_prediction_bs = prediction_result
                        last_prediction_num = final_num_pred
                        # Store for accuracy tracking
                        last_model_predictions_bs = {name: d.get('bs') for name, d in per_model_details.items() if d.get('bs') is not None}
                        last_model_predictions_num = {name: d.get('num') for name, d in per_model_details.items() if d.get('num') is not None}
                        last_model_predictions_col = {name: d.get('col') for name, d in per_model_details.items() if d.get('col') is not None}
                   

                    else:
                        print("⏭️ No predictions available this round")
                        last_prediction_bs = None

                    check_retrain_needed()
            else:
                print("⛔ OCR failed for this slot")
    except KeyboardInterrupt:
        print("\n🛑 Logger stopped by user")
        
if __name__ == "__main__":
    main()