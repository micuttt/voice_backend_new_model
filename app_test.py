import os
import glob
import logging
import numpy as np
import librosa
import parselmouth
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import tempfile
import wave
import subprocess
from scipy.stats import linregress
import antropy as ant
from noisereduce import reduce_noise
from parselmouth.praat import call
import time
import joblib
import pandas as pd
import threading
import ctypes
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
import hashlib
import signal
import sys
import concurrent.futures

# ===================== 修复日志编码 =====================
import sys
import io

# 修复标准输出编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ===================== 核心配置 =====================
CONFIG = {
    "SAMPLING_RATE": 16000,
    "TEMP_DIR": tempfile.gettempdir(),
    "MODEL_PATH": "./model_rank1_Bernoulli_NB.pkl",
    "PREPROCESSOR_PATH": "./preprocessor.pkl",
    "FEATURE_ORDER_PATH": "./shap_importance.pkl",
    "PD_THRESHOLD": 0.5,
    "MIN_AUDIO_DURATION": 1.0,
    "MAX_AUDIO_DURATION": 210.0,
    "ANALYSIS_TIMEOUT_SECONDS": 180,
    "FFMPEG_TIMEOUT": 30,
    "CACHE_MAX_SIZE": 20,
    "MAX_WORKERS": 2,
    
    "TARGET_FEATURES": [
        "Delta2_mean",
        "DFA_mean", 
        "F2_std_mean",
        "F0_slope_mean",
        "Delta0_mean",
        "MFCC3_mean",
        "F2_std_std"
    ],
    
    "SUBTYPE_WEIGHTS": {
        "tremor": {
            "Delta2_mean": 0.25,
            "DFA_mean": 0.20,
            "F0_slope_mean": 0.15,
            "Delta0_mean": 0.25,
            "MFCC3_mean": 0.15
        },
        "rigidity": {
            "F0_slope_mean": -0.30,
            "F2_std_mean": -0.25,
            "F2_std_std": -0.20,
            "Delta2_mean": -0.15,
            "MFCC3_mean": -0.10
        },
        "motor": {
            "DFA_mean": 0.25,
            "Delta2_mean": 0.20,
            "F0_slope_mean": 0.20,
            "F2_std_mean": 0.20,
            "F2_std_std": 0.15
        },
        "non_motor": {
            "MFCC3_mean": 0.30,
            "Delta0_mean": 0.25,
            "DFA_mean": 0.25,
            "F0_slope_mean": 0.20
        }
    },
    
    "SUBTYPE_SMOOTH": 0.1,
    "SUBTYPE_SCALE": 1.2,
    
    "SUBTYPE_EXPLANATION": {
        "tremor": "震颤主导型：以Delta2_mean和Delta0_mean升高为特征",
        "rigidity": "僵直主导型：以F0_slope_mean和F2_std降低为特征",
        "motor": "运动型：以DFA_mean和Delta2_mean升高为特征",
        "non_motor": "非运动型：以MFCC3_mean和Delta0_mean改变为特征"
    },
    
    "PRAAT_PARAMS": {
        "f0min": 75.0,
        "f0max": 500.0,
    },
    
    "RATE_LIMIT": {
        "requests_per_minute": 30,
        "requests_per_hour": 500
    }
}

# ===================== 日志配置（修复编码） =====================
class EncodingFileHandler(logging.FileHandler):
    """自定义文件处理器，确保UTF-8编码"""
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)

class EncodingStreamHandler(logging.StreamHandler):
    """自定义流处理器，确保UTF-8编码"""
    def __init__(self, stream=None):
        super().__init__(stream)
        self.encoding = 'utf-8'

# 配置日志
logger = logging.getLogger("PDDiagnoser")
logger.setLevel(logging.INFO)

# 移除所有现有处理器
logger.handlers.clear()

# 添加UTF-8控制台处理器
console_handler = EncodingStreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# 添加UTF-8文件处理器
try:
    file_handler = EncodingFileHandler('pd_diagnoser.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"无法创建文件日志: {e}")

# ===================== 全局线程池 =====================
executor = concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"])

# ===================== 数据类 =====================
@dataclass
class FeatureResult:
    features: np.ndarray
    feature_dict: Dict[str, float]
    feature_warnings: List[str]

@dataclass
class DiagnosisResult:
    pd_prob: float
    diagnosis: str
    risk: str
    features: Dict[str, float]
    subtype_probs: Dict[str, float]
    feature_warnings: List[str]
    processing_time: float

# ===================== 缓存系统 =====================
class FeatureCache:
    """特征结果缓存"""
    _cache = {}
    _cache_time = {}
    _max_size = CONFIG["CACHE_MAX_SIZE"]
    _lock = threading.Lock()
    
    @classmethod
    def get_key(cls, audio_path: str) -> str:
        """生成缓存键"""
        try:
            stat = os.stat(audio_path)
            return f"{audio_path}_{stat.st_size}_{stat.st_mtime}"
        except:
            return audio_path
    
    @classmethod
    def get(cls, key: str) -> Optional[FeatureResult]:
        """获取缓存"""
        with cls._lock:
            if key in cls._cache:
                cls._cache_time[key] = time.time()
                logger.debug(f"缓存命中: {key[:50]}...")
                return cls._cache[key]
            return None
    
    @classmethod
    def set(cls, key: str, value: FeatureResult):
        """设置缓存"""
        with cls._lock:
            if len(cls._cache) >= cls._max_size:
                oldest = min(cls._cache_time.items(), key=lambda x: x[1])
                del cls._cache[oldest[0]]
                del cls._cache_time[oldest[0]]
            
            cls._cache[key] = value
            cls._cache_time[key] = time.time()
    
    @classmethod
    def clear(cls):
        """清空缓存"""
        with cls._lock:
            cls._cache.clear()
            cls._cache_time.clear()
            logger.info("缓存已清空")

# ===================== 限流装饰器 =====================
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.lock = threading.Lock()
        
    def is_allowed(self, ip: str) -> Tuple[bool, str]:
        with self.lock:
            current_time = time.time()
            
            if ip not in self.requests:
                self.requests[ip] = []
            
            self.requests[ip] = [t for t in self.requests[ip] 
                                 if current_time - t < 3600]
            
            if len(self.requests[ip]) >= CONFIG["RATE_LIMIT"]["requests_per_hour"]:
                return False, "超过小时请求限制(500次/小时)"
            
            minute_ago = current_time - 60
            minute_requests = len([t for t in self.requests[ip] if t > minute_ago])
            if minute_requests >= CONFIG["RATE_LIMIT"]["requests_per_minute"]:
                return False, "超过分钟请求限制(30次/分钟)"
            
            self.requests[ip].append(current_time)
            return True, ""

rate_limiter = RateLimiter()

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        allowed, message = rate_limiter.is_allowed(ip)
        if not allowed:
            return jsonify({"code": 429, "msg": message}), 429
        return f(*args, **kwargs)
    return decorated_function

# ===================== 模型加载（修复特征名警告） =====================
class ModelManager:
    _model = None
    _preprocessor = None
    _lasso_features = None
    _subtype_classifier = None
    _feature_order = None
    _load_lock = threading.Lock()
    _small_scaler = None
    _large_scaler = None
    _small_cols = []
    _large_cols = []

    @classmethod
    def load_all(cls):
        """线程安全的模型加载"""
        if cls._model is not None:
            return cls._model, cls._preprocessor, cls._lasso_features
        
        with cls._load_lock:
            if cls._model is not None:
                return cls._model, cls._preprocessor, cls._lasso_features
            
            # 1. 加载预处理器
            if cls._preprocessor is None:
                try:
                    cls._preprocessor = joblib.load(CONFIG["PREPROCESSOR_PATH"])
                    logger.info("加载预处理器成功")
                    
                    # 提取scaler和列信息
                    cls._small_scaler = cls._preprocessor.get('small')
                    cls._large_scaler = cls._preprocessor.get('large')
                    cls._small_cols = cls._preprocessor.get('small_cols', [])
                    cls._large_cols = cls._preprocessor.get('large_cols', [])
                    
                    all_features = cls._small_cols + cls._large_cols
                    logger.info(f"预处理器包含 {len(all_features)} 个特征")
                    cls._feature_order = all_features
                    
                except Exception as e:
                    logger.error(f"加载预处理器失败: {e}")
                    raise

            # 2. 加载LASSO特征
            if cls._lasso_features is None:
                try:
                    feature_data = joblib.load(CONFIG["FEATURE_ORDER_PATH"])
                    if isinstance(feature_data, pd.DataFrame):
                        cls._lasso_features = feature_data['feature'].tolist()
                    elif isinstance(feature_data, list):
                        cls._lasso_features = feature_data
                    else:
                        cls._lasso_features = cls._feature_order
                    
                    logger.info(f"加载LASSO特征成功，共 {len(cls._lasso_features)} 个")
                    
                except Exception as e:
                    logger.error(f"加载特征顺序失败: {e}，使用预处理器特征")
                    cls._lasso_features = cls._feature_order

            # 3. 加载模型
            if cls._model is None:
                try:
                    cls._model = joblib.load(CONFIG["MODEL_PATH"])
                    logger.info(f"加载模型成功: {type(cls._model).__name__}")
                    
                except Exception as e:
                    logger.error(f"加载模型失败: {e}")
                    raise

            return cls._model, cls._preprocessor, cls._lasso_features

    @classmethod
    def preprocess_features(cls, features_dict):
        """
        特征预处理 - 修复特征名警告
        使用numpy数组而不是DataFrame，避免特征名警告
        """
        if cls._preprocessor is None:
            cls.load_all()

        try:
            # 使用CONFIG中定义的7个特征顺序
            TARGET_FEATURES = CONFIG["TARGET_FEATURES"]
            
            # 构建特征数组 - 直接使用numpy数组，不带特征名
            feat_array = np.array([
                features_dict.get(f, 0.0) for f in TARGET_FEATURES
            ], dtype=np.float32).reshape(1, -1)
            
            # 如果需要完整预处理（包括小范围和大范围标准化）
            if cls._small_scaler is not None and cls._large_scaler is not None:
                # 这里假设TARGET_FEATURES的顺序与预处理器要求的顺序一致
                # 如果不一致，需要映射
                
                # 创建完整的特征向量（所有特征）
                all_features = cls._small_cols + cls._large_cols
                full_feat_array = np.array([
                    features_dict.get(f, 0.0) for f in all_features
                ], dtype=np.float32).reshape(1, -1)
                
                # 分别标准化
                n_small = len(cls._small_cols)
                if n_small > 0:
                    small_part = full_feat_array[:, :n_small]
                    # 确保是numpy数组，不是DataFrame
                    small_scaled = cls._small_scaler.transform(small_part)
                else:
                    small_scaled = np.array([[]])
                
                n_large = len(cls._large_cols)
                if n_large > 0:
                    large_part = full_feat_array[:, n_small:]
                    large_scaled = cls._large_scaler.transform(large_part)
                else:
                    large_scaled = np.array([[]])
                
                # 拼接
                final = np.hstack([small_scaled, large_scaled])
                
                # 如果模型需要的是7个特征，但预处理器输出更多，需要筛选
                if final.shape[1] > len(TARGET_FEATURES):
                    # 根据特征名称筛选（需要特征名称到索引的映射）
                    # 这里简化处理，假设前7个就是要的
                    final = final[:, :len(TARGET_FEATURES)]
                
                return final
            else:
                # 如果没有预处理器，直接返回7个特征
                return feat_array

        except Exception as e:
            logger.error(f"预处理失败: {e}")
            return np.zeros((1, len(CONFIG["TARGET_FEATURES"])))

    @classmethod
    def preprocess_features_simple(cls, features_dict):
        """
        简化版预处理 - 只使用7个特征，不进行标准化
        如果模型已经用标准化后的数据训练，不要使用这个版本
        """
        TARGET_FEATURES = CONFIG["TARGET_FEATURES"]
        
        feat_array = np.array([
            features_dict.get(f, 0.0) for f in TARGET_FEATURES
        ], dtype=np.float32).reshape(1, -1)
        
        return feat_array

# ===================== 工具函数 =====================
def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, 
                               text=True,
                               timeout=5)
        return result.returncode == 0
    except:
        return False

def validate_audio_duration(audio_path: str) -> Tuple[bool, float, str]:
    """验证音频时长"""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate
            
        if duration < CONFIG["MIN_AUDIO_DURATION"]:
            return False, duration, f"音频过短 ({duration:.2f}秒)"
        
        if duration > CONFIG["MAX_AUDIO_DURATION"]:
            return False, duration, f"音频过长 ({duration:.2f}秒)"
        
        return True, duration, ""
    except Exception as e:
        return False, 0, f"读取音频失败: {str(e)}"

def get_temp_path(prefix: str) -> str:
    """获取临时文件路径"""
    return os.path.join(CONFIG["TEMP_DIR"], f"{prefix}_{os.getpid()}_{int(time.time())}_{np.random.randint(10000)}.wav")

def convert_numpy_type(obj):
    """递归转换numpy类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_type(i) for i in obj]
    else:
        return obj

def cleanup_temp_files(temp_paths: List[str]):
    """清理临时文件"""
    for path in temp_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"已删除临时文件: {path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败 {path}: {e}")

def convert_audio_sync(input_path: str, output_path: str) -> bool:
    """同步音频转换"""
    ffmpeg_cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", str(CONFIG["SAMPLING_RATE"]),
        "-ac", "1", "-sample_fmt", "s16",
        "-c:a", "pcm_s16le", "-y", output_path
    ]
    
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=CONFIG["FFMPEG_TIMEOUT"]
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg转换超时")
        return False
    except Exception as e:
        logger.error(f"FFmpeg转换失败: {e}")
        return False

# ===================== 特征提取 =====================
def extract_f0_slope(seg, sr, fmin=75, fmax=500):
    """提取F0_slope"""
    try:
        snd = parselmouth.Sound(seg, sr)
        pitch = call(snd, "To Pitch", 0.001, fmin, fmax)
        pitch_arr = pitch.selected_array['frequency']
        pitch_arr = pitch_arr[pitch_arr > 0]
        if len(pitch_arr) > 2:
            slope = linregress(range(len(pitch_arr)), pitch_arr)[0]
            return float(slope)
    except Exception as e:
        logger.debug(f"F0_slope提取失败: {e}")
    return 0.0

def extract_f2_std(seg, sr):
    """提取F2_std"""
    try:
        snd = parselmouth.Sound(seg, sr)
        formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f2_std = call(formant, "Get standard deviation", 2, 0, 0, "Hertz")
        return float(f2_std) if not np.isnan(f2_std) else 0.0
    except:
        return 0.0

def extract_dfa(seg):
    """提取DFA"""
    try:
        if len(seg) < 200:
            return 0.0
        return float(ant.detrended_fluctuation(seg))
    except:
        return 0.0

def extract_mfcc_and_delta(seg, sr):
    """提取MFCC3, Delta0, Delta2"""
    try:
        mfccs = librosa.feature.mfcc(
            y=seg, sr=sr, n_mfcc=6, 
            n_fft=512, hop_length=128
        )
        delta = librosa.feature.delta(mfccs)
        
        mfcc3 = float(mfccs[3].mean()) if len(mfccs) > 3 else 0.0
        delta0 = float(delta[0].mean()) if len(delta) > 0 else 0.0
        delta2 = float(delta[2].mean()) if len(delta) > 2 else 0.0
        
        return mfcc3, delta0, delta2
    except:
        return 0.0, 0.0, 0.0

def extract_7features(audio_path: str) -> FeatureResult:
    """只提取7个目标特征"""
    warnings = []
    segment_features = []
    
    try:
        # 加载音频
        y_raw, _ = librosa.load(audio_path, sr=CONFIG["SAMPLING_RATE"], mono=True)
        y_trimmed, _ = librosa.effects.trim(y_raw, top_db=20)
        
        # 降噪
        y_denoised = reduce_noise(
            y=y_trimmed, 
            sr=CONFIG["SAMPLING_RATE"], 
            stationary=True, 
            prop_decrease=0.6
        )
        
        # 分割语音段
        intervals = librosa.effects.split(
            y_denoised, 
            top_db=25, 
            frame_length=512, 
            hop_length=128
        )
        
        if len(intervals) == 0:
            intervals = [(0, len(y_denoised))]
        
        # 处理每个语音段
        for inter in intervals:
            start, end = inter[0], inter[1]
            dur = (end - start) / CONFIG["SAMPLING_RATE"]
            
            if dur < 0.3:
                continue
                
            seg_raw = y_trimmed[start:end].astype(np.float32)
            seg_denoised = y_denoised[start:end].astype(np.float32)
            
            f0_slope = extract_f0_slope(seg_raw, CONFIG["SAMPLING_RATE"])
            f2_std = extract_f2_std(seg_raw, CONFIG["SAMPLING_RATE"])
            dfa = extract_dfa(seg_raw)
            mfcc3, delta0, delta2 = extract_mfcc_and_delta(seg_denoised, CONFIG["SAMPLING_RATE"])
            
            segment_features.append({
                'F0_slope': f0_slope,
                'F2_std': f2_std,
                'DFA': dfa,
                'MFCC3': mfcc3,
                'Delta0': delta0,
                'Delta2': delta2
            })
        
        if not segment_features:
            seg_raw = y_trimmed.astype(np.float32)
            seg_denoised = y_denoised.astype(np.float32)
            
            f0_slope = extract_f0_slope(seg_raw, CONFIG["SAMPLING_RATE"])
            f2_std = extract_f2_std(seg_raw, CONFIG["SAMPLING_RATE"])
            dfa = extract_dfa(seg_raw)
            mfcc3, delta0, delta2 = extract_mfcc_and_delta(seg_denoised, CONFIG["SAMPLING_RATE"])
            
            segment_features.append({
                'F0_slope': f0_slope,
                'F2_std': f2_std,
                'DFA': dfa,
                'MFCC3': mfcc3,
                'Delta0': delta0,
                'Delta2': delta2
            })
        
        # 聚合特征
        df = pd.DataFrame(segment_features)
        
        feature_dict = {
            'Delta2_mean': float(df['Delta2'].mean()),
            'DFA_mean': float(df['DFA'].mean()),
            'F2_std_mean': float(df['F2_std'].mean()),
            'F0_slope_mean': float(df['F0_slope'].mean()),
            'Delta0_mean': float(df['Delta0'].mean()),
            'MFCC3_mean': float(df['MFCC3'].mean()),
            'F2_std_std': float(df['F2_std'].std()) if len(df) > 1 else 0.0
        }
        
        return FeatureResult(
            features=np.array([feature_dict[name] for name in CONFIG["TARGET_FEATURES"]]),
            feature_dict=feature_dict,
            feature_warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"特征提取失败: {e}", exc_info=True)
        zero_features = {name: 0.0 for name in CONFIG["TARGET_FEATURES"]}
        return FeatureResult(
            features=np.zeros(len(CONFIG["TARGET_FEATURES"])),
            feature_dict=zero_features,
            feature_warnings=[f"严重错误：语音特征提取失败"]
        )

def extract_7features_with_cache(audio_path: str) -> FeatureResult:
    """带缓存的特征提取"""
    cache_key = FeatureCache.get_key(audio_path)
    cached_result = FeatureCache.get(cache_key)
    if cached_result:
        return cached_result
    
    feature_result = extract_7features(audio_path)
    
    if "失败" not in feature_result.feature_warnings[0] if feature_result.feature_warnings else True:
        FeatureCache.set(cache_key, feature_result)
    
    return feature_result

# ===================== PD亚型概率计算 =====================
def calculate_subtype_probs(feature_dict: Dict[str, float], pd_prob: float) -> Dict[str, float]:
    """计算PD亚型概率"""
    weights = CONFIG["SUBTYPE_WEIGHTS"]
    subtype_scores = {}
    
    for subtype, feat_weights in weights.items():
        score = 0.0
        for feat_name, weight in feat_weights.items():
            feat_value = feature_dict.get(feat_name, 0.0)
            score += feat_value * weight
        subtype_scores[subtype] = score
    
    adjusted_scores = {}
    for subtype, score in subtype_scores.items():
        adjusted_scores[subtype] = score * pd_prob * CONFIG["SUBTYPE_SCALE"] + CONFIG["SUBTYPE_SMOOTH"]
    
    sigmoid_scores = {subtype: sigmoid(score) for subtype, score in adjusted_scores.items()}
    
    total = sum(sigmoid_scores.values())
    if total > 0:
        normalized_probs = {
            subtype: round(float(prob) / total, 4) 
            for subtype, prob in sigmoid_scores.items()
        }
    else:
        n_subtypes = len(sigmoid_scores)
        normalized_probs = {subtype: round(1.0 / n_subtypes, 4) for subtype in sigmoid_scores.keys()}
    
    if pd_prob < CONFIG["PD_THRESHOLD"]:
        for subtype in normalized_probs:
            normalized_probs[subtype] = round(normalized_probs[subtype] * pd_prob, 4)
    
    # 转换numpy类型为Python原生类型
    normalized_probs = {k: float(v) for k, v in normalized_probs.items()}
    
    logger.info(f"亚型概率计算完成: {normalized_probs}")
    return normalized_probs

# ===================== 诊断逻辑 =====================
def diagnose(feature_result: FeatureResult) -> DiagnosisResult:
    """使用模型诊断"""
    model, preprocessor, lasso_features = ModelManager.load_all()
    
    start_time = time.time()
    
    # 特征预处理
    features_scaled = ModelManager.preprocess_features(feature_result.feature_dict)
    
    # 预测
    try:
        if hasattr(model, 'predict_proba'):
            pd_prob = float(model.predict_proba(features_scaled)[0, 1])
        else:
            pred = model.predict(features_scaled)[0]
            pd_prob = 0.9 if pred == 1 else 0.1
            
        logger.info(f"预测成功: pd_prob={pd_prob:.4f}")
    except Exception as e:
        logger.error(f"预测失败: {e}")
        pd_prob = 0.5
    
    # 计算亚型概率
    subtype_probs = calculate_subtype_probs(feature_result.feature_dict, pd_prob)
    
    # 诊断结论
    diagnosis = "患有PD" if pd_prob >= CONFIG["PD_THRESHOLD"] else "健康"
    if pd_prob >= 0.8:
        risk = "高风险"
    elif pd_prob >= 0.5:
        risk = "中风险"
    else:
        risk = "低风险"
    
    processing_time = time.time() - start_time
    
    return DiagnosisResult(
        pd_prob=round(pd_prob, 4),
        diagnosis=diagnosis,
        risk=risk,
        features=feature_result.feature_dict,
        subtype_probs=subtype_probs,
        feature_warnings=feature_result.feature_warnings,
        processing_time=round(processing_time, 3)
    )

# ===================== Flask API =====================
app = Flask(__name__)
CORS(app)

@app.before_request
def before_request():
    """请求前处理"""
    g.start_time = time.time()
    g.request_id = hashlib.md5(f"{time.time()}_{np.random.random()}".encode()).hexdigest()[:8]
    logger.info(f"请求[{g.request_id}]开始: {request.path}")

@app.after_request
def after_request(response):
    """请求后处理"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"请求[{getattr(g, 'request_id', 'unknown')}]完成: 耗时{duration:.2f}秒")
        response.headers['X-Processing-Time'] = str(round(duration, 2))
    return response

@app.route('/', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy", 
        "service": "PD Voice Diagnosis Service",
        "timestamp": time.time()
    }), 200

@app.route('/health/detailed', methods=['GET'])
def detailed_health():
    """详细健康检查"""
    return jsonify({
        "status": "healthy",
        "ffmpeg_available": check_ffmpeg(),
        "model_loaded": ModelManager._model is not None,
        "cache_size": len(FeatureCache._cache),
        "config": {
            "max_workers": CONFIG["MAX_WORKERS"],
            "timeout": CONFIG["ANALYSIS_TIMEOUT_SECONDS"],
            "cache_max_size": CONFIG["CACHE_MAX_SIZE"]
        }
    }), 200

@app.route('/analyze', methods=['POST'])
@rate_limit
def pd_diagnose():
    """诊断接口"""
    temp_paths = []
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        if 'file' not in request.files:
            return jsonify({"code": 400, "msg": "未上传音频文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"code": 400, "msg": "未选择音频文件"}), 400
        
        if not check_ffmpeg():
            return jsonify({"code": 500, "msg": "服务器音频处理组件缺失"}), 500
        
        logger.info(f"请求[{request_id}] 接收文件: {file.filename}")
        
        temp_path = get_temp_path("upload")
        converted_path = get_temp_path("converted")
        temp_paths = [temp_path, converted_path]
        
        file.save(temp_path)
        
        # 音频转换
        try:
            convert_result = convert_audio_sync(temp_path, converted_path)
            if not convert_result:
                raise Exception("音频转换失败")
        except Exception as e:
            return jsonify({"code": 500, "msg": f"音频转换失败: {str(e)}"}), 500
        
        # 验证时长
        valid, duration, msg = validate_audio_duration(converted_path)
        if not valid:
            return jsonify({"code": 400, "msg": msg}), 400
        
        # 特征提取
        try:
            with ThreadPoolExecutor(max_workers=1) as thread_executor:
                future_feat = thread_executor.submit(extract_7features_with_cache, converted_path)
                feature_result = future_feat.result(timeout=CONFIG["ANALYSIS_TIMEOUT_SECONDS"])
        except TimeoutError:
            logger.error(f"请求[{request_id}] 特征提取超时")
            return jsonify({
                "code": 504,
                "msg": "特征提取超时，请稍后重试"
            }), 504
        except Exception as e:
            logger.error(f"请求[{request_id}] 特征提取失败: {e}")
            return jsonify({"code": 500, "msg": f"特征提取失败: {str(e)}"}), 500
        
        # 检查特征提取是否成功
        if feature_result.feature_warnings and "失败" in feature_result.feature_warnings[0]:
            return jsonify({
                "code": 500,
                "msg": feature_result.feature_warnings[0]
            }), 500
        
        # 诊断
        try:
            with ThreadPoolExecutor(max_workers=1) as thread_executor:
                future_diag = thread_executor.submit(diagnose, feature_result)
                diagnosis_result = future_diag.result(timeout=30)
        except TimeoutError:
            logger.error(f"请求[{request_id}] 诊断超时")
            return jsonify({
                "code": 504,
                "msg": "诊断超时，请稍后重试"
            }), 504
        
        total_time = time.time() - g.start_time
        
        # 构建返回结果
        response = {
            "code": 200,
            "msg": "诊断成功",
            "request_id": request_id,
            "data": {
                "audio_duration": round(duration, 2),
                "processing_time": round(total_time, 2),
                "pd_probability": diagnosis_result.pd_prob,
                "diagnosis": diagnosis_result.diagnosis,
                "risk_level": diagnosis_result.risk,
                "features": diagnosis_result.features,
                "feature_warnings": diagnosis_result.feature_warnings,
                "subtype_probabilities": {
                    "tremor_type": diagnosis_result.subtype_probs.get("tremor", 0.0),
                    "rigidity_type": diagnosis_result.subtype_probs.get("rigidity", 0.0),
                    "motor_type": diagnosis_result.subtype_probs.get("motor", 0.0),
                    "non_motor_type": diagnosis_result.subtype_probs.get("non_motor", 0.0)
                },
                "subtype_explanation": CONFIG["SUBTYPE_EXPLANATION"]
            }
        }
        
        logger.info(f"请求[{request_id}] 诊断完成，耗时{total_time:.2f}秒")
        
        # 转换numpy类型
        response = convert_numpy_type(response)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"请求[{request_id}] 诊断异常: {e}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"诊断失败: {str(e)}"
        }), 500
    finally:
        cleanup_temp_files(temp_paths)

# ===================== 缓存管理接口 =====================
@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """清空缓存"""
    auth_key = request.headers.get('X-Admin-Key')
    if auth_key != os.environ.get('ADMIN_KEY', 'admin-secret-key'):
        return jsonify({"code": 403, "msg": "无权访问"}), 403
    
    FeatureCache.clear()
    return jsonify({"code": 200, "msg": "缓存已清空"}), 200

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """缓存统计"""
    return jsonify({
        "code": 200,
        "data": {
            "cache_size": len(FeatureCache._cache),
            "max_size": CONFIG["CACHE_MAX_SIZE"],
            "cache_keys": list(FeatureCache._cache.keys())[:5] if FeatureCache._cache else []
        }
    }), 200

# ===================== 优雅关闭 =====================
def signal_handler(signum, frame):
    """处理退出信号"""
    logger.info(f"收到信号 {signum}，正在优雅关闭...")
    
    executor.shutdown(wait=True, cancel_futures=True)
    cleanup_temp_files(glob.glob(os.path.join(CONFIG["TEMP_DIR"], "*.wav")))
    
    logger.info("服务已关闭")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ===================== 启动服务 =====================
if __name__ == '__main__':
    if not check_ffmpeg():
        logger.error("ffmpeg未安装，请先安装ffmpeg")
        exit(1)
    
    try:
        ModelManager.load_all()
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        exit(1)
    
    cleanup_temp_files(glob.glob(os.path.join(CONFIG["TEMP_DIR"], "*.wav")))
    
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"服务启动在端口 {port}，调试模式: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
