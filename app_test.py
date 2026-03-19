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
import json
import gc
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
    "MODEL_PATH": "./model_rank1_Logistic_Regression(slight).pkl",
    "PREPROCESSOR_PATH": "./preprocessor(slight).pkl",
    "FEATURE_ORDER_PATH": "./shap_importance(slight).pkl",
    "PD_THRESHOLD": 0.5,
    "MIN_AUDIO_DURATION": 1.0,
    "MAX_AUDIO_DURATION": 210.0,
    "ANALYSIS_TIMEOUT_SECONDS": 180,
    "FFMPEG_TIMEOUT": 30,
    "CACHE_MAX_SIZE": 3,
    "MAX_WORKERS": 1,
    
    # ⚠️ 新的7个目标特征（严格按照LASSO筛选后的顺序）
    "TARGET_FEATURES": [
        "F0_max_std",      # 索引0
        "Delta0_mean",      # 索引1
        "Delta2_mean",      # 索引2
        "F0_slope_mean",    # 索引3
        "F2_std_mean",      # 索引4
        "DFA_mean",         # 索引5
        "MFCC4_mean"        # 索引6
    ],
    
    "SUBTYPE_WEIGHTS": {
        "tremor": {
            "Delta2_mean": 0.25,
            "DFA_mean": 0.20,
            "F0_slope_mean": 0.15,
            "Delta0_mean": 0.25,
            "MFCC4_mean": 0.15
        },
        "rigidity": {
            "F0_slope_mean": -0.30,
            "F2_std_mean": -0.25,
            "F0_max_std": -0.20,
            "Delta2_mean": -0.15,
            "MFCC4_mean": -0.10
        },
        "motor": {
            "DFA_mean": 0.25,
            "Delta2_mean": 0.20,
            "F0_slope_mean": 0.20,
            "F2_std_mean": 0.20,
            "F0_max_std": 0.15
        },
        "non_motor": {
            "MFCC4_mean": 0.30,
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
        "non_motor": "非运动型：以MFCC4_mean和Delta0_mean改变为特征"
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

# ===================== 日志配置（增强版） =====================
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
logger.setLevel(logging.DEBUG)  # 改为DEBUG级别以获取更多信息

# 移除所有现有处理器
logger.handlers.clear()

# 添加UTF-8控制台处理器
console_handler = EncodingStreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)



# ===================== 全局线程池 =====================
executor = concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"])

# ===================== 数据类 =====================
@dataclass
class FeatureResult:
    features: np.ndarray
    feature_dict: Dict[str, float]
    feature_warnings: List[str]
    raw_segment_features: List[Dict] = None  # 新增：保存原始分段特征用于调试

@dataclass
class DiagnosisResult:
    pd_prob: float
    diagnosis: str
    risk: str
    features: Dict[str, float]
    subtype_probs: Dict[str, float]
    feature_warnings: List[str]
    processing_time: float
    raw_features_scaled: np.ndarray = None  # 新增：保存标准化后的特征

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

# ===================== 模型加载（修复特征名警告 + 详细日志） =====================
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
    _feature_to_index = {}  # 新增：特征名到索引的映射

    @classmethod
    def load_all(cls):
        """线程安全的模型加载（增强日志）"""
        if cls._model is not None:
            return cls._model, cls._preprocessor, cls._lasso_features
        
        with cls._load_lock:
            if cls._model is not None:
                return cls._model, cls._preprocessor, cls._lasso_features
            
            # 1. 加载预处理器
            if cls._preprocessor is None:
                try:
                    logger.info(f"正在加载预处理器: {CONFIG['PREPROCESSOR_PATH']}")
                    cls._preprocessor = joblib.load(CONFIG["PREPROCESSOR_PATH"])
                    
                    # 提取scaler和列信息
                    cls._small_scaler = cls._preprocessor.get('small')
                    cls._large_scaler = cls._preprocessor.get('large')
                    cls._small_cols = cls._preprocessor.get('small_cols', [])
                    cls._large_cols = cls._preprocessor.get('large_cols', [])
                    
                    all_features = cls._small_cols + cls._large_cols
                    logger.info(f"预处理器包含 {len(all_features)} 个特征")
                    logger.debug(f"小范围特征({len(cls._small_cols)}): {cls._small_cols}")
                    logger.debug(f"大范围特征({len(cls._large_cols)}): {cls._large_cols}")
                    
                    # 构建特征名到索引的映射
                    cls._feature_to_index = {name: idx for idx, name in enumerate(all_features)}
                    cls._feature_order = all_features
                    
                except Exception as e:
                    logger.error(f"加载预处理器失败: {e}", exc_info=True)
                    raise

            # 2. 加载LASSO特征（特征顺序）
            if cls._lasso_features is None:
                try:
                    logger.info(f"正在加载特征顺序: {CONFIG['FEATURE_ORDER_PATH']}")
                    feature_data = joblib.load(CONFIG["FEATURE_ORDER_PATH"])
                    
                    if isinstance(feature_data, pd.DataFrame):
                        if 'feature' in feature_data.columns:
                            cls._lasso_features = feature_data['feature'].tolist()
                        else:
                            cls._lasso_features = feature_data.iloc[:, 0].tolist()
                    elif isinstance(feature_data, list):
                        cls._lasso_features = feature_data
                    else:
                        logger.warning("特征顺序格式未知，使用预处理器特征")
                        cls._lasso_features = cls._feature_order
                    
                    logger.info(f"加载LASSO特征成功，共 {len(cls._lasso_features)} 个")
                    logger.info(f"LASSO特征顺序: {cls._lasso_features}")
                    
                    # 验证特征是否都在预处理器的特征中
                    missing_features = [f for f in cls._lasso_features if f not in cls._feature_to_index]
                    if missing_features:
                        logger.error(f"LASSO特征不在预处理器中: {missing_features}")
                    
                except Exception as e:
                    logger.error(f"加载特征顺序失败: {e}，使用预处理器特征", exc_info=True)
                    cls._lasso_features = cls._feature_order

            # 3. 加载模型
            if cls._model is None:
                try:
                    logger.info(f"正在加载模型: {CONFIG['MODEL_PATH']}")
                    cls._model = joblib.load(CONFIG["MODEL_PATH"])
                    logger.info(f"加载模型成功: {type(cls._model).__name__}")
                    
                    
                except Exception as e:
                    logger.error(f"加载模型失败: {e}", exc_info=True)
                    raise

            return cls._model, cls._preprocessor, cls._lasso_features

    @classmethod
    def preprocess_features(cls, features_dict, request_id="unknown"):
        """
        特征预处理 - 简化版：直接处理7个特征
        严格按照CONFIG["TARGET_FEATURES"]的顺序
        """
        if cls._preprocessor is None:
            cls.load_all()

        try:
            # 记录原始特征值
            logger.debug(f"请求[{request_id}] 原始特征字典: {json.dumps({k: f'{v:.4f}' for k, v in features_dict.items()})}")
            
            # 步骤1: 按照固定顺序构建7维特征数组
            TARGET_FEATURES = CONFIG["TARGET_FEATURES"]
            
            # 检查是否有缺失特征
            missing_in_dict = [f for f in TARGET_FEATURES if f not in features_dict]
            if missing_in_dict:
                logger.warning(f"请求[{request_id}] 特征字典中缺失: {missing_in_dict}")
            
            # 构建7维特征数组（严格按顺序）
            raw_features = []
            for feat_name in TARGET_FEATURES:
                val = features_dict.get(feat_name, 0.0)
                raw_features.append(val)
            
            # 转换为numpy数组 (1, 7)
            raw_features = np.array(raw_features, dtype=np.float32).reshape(1, -1)
            logger.debug(f"请求[{request_id}] 原始7特征: {raw_features[0].tolist()}")
            
            # 步骤2: 分别标准化（根据预处理器中的分类）
            small_cols = cls._small_cols  # 小范围特征列表
            large_cols = cls._large_cols  # 大范围特征列表
            
            # 创建副本用于标准化
            scaled_features = raw_features.copy()
            
            # 处理小范围特征（StandardScaler）
            if small_cols and cls._small_scaler is not None:
                # 找到小范围特征在7维数组中的索引
                small_indices = [TARGET_FEATURES.index(col) for col in small_cols if col in TARGET_FEATURES]
                if small_indices:
                    logger.debug(f"请求[{request_id}] 小范围特征索引: {small_indices}")
                    # 提取对应列进行标准化
                    small_part = raw_features[:, small_indices]
                    small_scaled = cls._small_scaler.transform(small_part)
                    scaled_features[:, small_indices] = small_scaled
                    logger.debug(f"请求[{request_id}] 小范围特征标准化后: {small_scaled[0].tolist()}")
            
            # 处理大范围特征（RobustScaler）
            if large_cols and cls._large_scaler is not None:
                # 找到大范围特征在7维数组中的索引
                large_indices = [TARGET_FEATURES.index(col) for col in large_cols if col in TARGET_FEATURES]
                if large_indices:
                    logger.debug(f"请求[{request_id}] 大范围特征索引: {large_indices}")
                    # 提取对应列进行标准化
                    large_part = raw_features[:, large_indices]
                    large_scaled = cls._large_scaler.transform(large_part)
                    scaled_features[:, large_indices] = large_scaled
                    logger.debug(f"请求[{request_id}] 大范围特征标准化后: {large_scaled[0].tolist()}")
            
            logger.debug(f"请求[{request_id}] 最终标准化后的7特征: {scaled_features[0].tolist()}")
            
            return scaled_features

        except Exception as e:
            logger.error(f"请求[{request_id}] 预处理失败: {e}", exc_info=True)
            return np.zeros((1, len(CONFIG["TARGET_FEATURES"])))

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
                               timeout=50)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"FFmpeg检查失败: {e}")
        return False

def validate_audio_duration(audio_path: str) -> Tuple[bool, float, str]:
    """验证音频时长"""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate
            
        logger.debug(f"音频时长: {duration:.2f}秒, 采样率: {rate}Hz")
        
        if duration < CONFIG["MIN_AUDIO_DURATION"]:
            return False, duration, f"音频过短 ({duration:.2f}秒)"
        
        if duration > CONFIG["MAX_AUDIO_DURATION"]:
            return False, duration, f"音频过长 ({duration:.2f}秒)"
        
        return True, duration, ""
    except Exception as e:
        logger.error(f"读取音频失败: {e}", exc_info=True)
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

def convert_audio_sync(input_path: str, output_path: str, request_id="unknown") -> bool:
    """同步音频转换"""
    ffmpeg_cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", str(CONFIG["SAMPLING_RATE"]),
        "-ac", "1", "-sample_fmt", "s16",
        "-c:a", "pcm_s16le", "-y", output_path
    ]
    
    logger.debug(f"请求[{request_id}] FFmpeg命令: {' '.join(ffmpeg_cmd)}")
    
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=CONFIG["FFMPEG_TIMEOUT"]
        )
        
        if result.returncode != 0:
            logger.error(f"请求[{request_id}] FFmpeg转换失败: {result.stderr}")
            return False
            
        logger.debug(f"请求[{request_id}] FFmpeg转换成功")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"请求[{request_id}] FFmpeg转换超时")
        return False
    except Exception as e:
        logger.error(f"请求[{request_id}] FFmpeg转换异常: {e}", exc_info=True)
        return False

# ===================== 修正后的特征提取函数 =====================

def extract_f0_max(seg, sr, fmin=75, fmax=500, request_id="unknown", segment_idx=0):
    """提取F0_max：基频最大值"""
    try:
        snd = parselmouth.Sound(seg, sr)
        pitch = call(snd, "To Pitch", 0.001, fmin, fmax)
        f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        result = float(f0_max) if not np.isnan(f0_max) else 0.0
        logger.debug(f"请求[{request_id}] 分段{segment_idx} F0_max={result:.4f}")
        return result
    except Exception as e:
        logger.debug(f"请求[{request_id}] 分段{segment_idx} F0_max提取失败: {e}")
        return 0.0

def extract_f0_slope(seg, sr, fmin=75, fmax=500, request_id="unknown", segment_idx=0):
    """提取F0_slope：基频斜率"""
    try:
        snd = parselmouth.Sound(seg, sr)
        pitch = call(snd, "To Pitch", 0.001, fmin, fmax)
        pitch_arr = pitch.selected_array['frequency']
        pitch_arr = pitch_arr[pitch_arr > 0]
        if len(pitch_arr) > 2:
            slope = linregress(range(len(pitch_arr)), pitch_arr)[0]
            logger.debug(f"请求[{request_id}] 分段{segment_idx} F0_slope={slope:.4f}")
            return float(slope)
        else:
            logger.debug(f"请求[{request_id}] 分段{segment_idx} 有效基频点不足: {len(pitch_arr)}")
            return 0.0
    except Exception as e:
        logger.debug(f"请求[{request_id}] 分段{segment_idx} F0_slope提取失败: {e}")
        return 0.0

def extract_f2_std(seg, sr, request_id="unknown", segment_idx=0):
    """提取F2_std：第二共振峰标准差"""
    try:
        snd = parselmouth.Sound(seg, sr)
        formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f2_std = call(formant, "Get standard deviation", 2, 0, 0, "Hertz")
        result = float(f2_std) if not np.isnan(f2_std) else 0.0
        logger.debug(f"请求[{request_id}] 分段{segment_idx} F2_std={result:.4f}")
        return result
    except Exception as e:
        logger.debug(f"请求[{request_id}] 分段{segment_idx} F2_std提取失败: {e}")
        return 0.0

def extract_dfa(seg, request_id="unknown", segment_idx=0):
    """提取DFA：去趋势波动分析"""
    try:
        if len(seg) < 200:
            logger.debug(f"请求[{request_id}] 分段{segment_idx} 信号太短({len(seg)}), 无法计算DFA")
            return 0.0
        result = float(ant.detrended_fluctuation(seg))
        logger.debug(f"请求[{request_id}] 分段{segment_idx} DFA={result:.4f}")
        return result
    except Exception as e:
        logger.debug(f"请求[{request_id}] 分段{segment_idx} DFA提取失败: {e}")
        return 0.0

def extract_mfcc_and_delta(seg, sr, request_id="unknown", segment_idx=0):
    """
    提取MFCC和Delta特征
    返回: (MFCC4, Delta0, Delta2) 注意这里是每个语音段的原始值，不是均值
    """
    try:
        mfccs = librosa.feature.mfcc(
            y=seg, sr=sr, n_mfcc=5,  # 需要MFCC4，所以至少5个
            n_fft=512, hop_length=128
        )
        delta = librosa.feature.delta(mfccs)
        
        mfcc4 = float(mfccs[4].mean()) if len(mfccs) > 4 else 0.0  # MFCC4的均值
        delta0 = float(delta[0].mean()) if len(delta) > 0 else 0.0  # Delta0的均值
        delta2 = float(delta[2].mean()) if len(delta) > 2 else 0.0  # Delta2的均值
        
        logger.debug(f"请求[{request_id}] 分段{segment_idx} MFCC4={mfcc4:.4f}, Delta0={delta0:.4f}, Delta2={delta2:.4f}")
        return mfcc4, delta0, delta2
    except Exception as e:
        logger.debug(f"请求[{request_id}] 分段{segment_idx} MFCC/Delta提取失败: {e}")
        return 0.0, 0.0, 0.0

def extract_7features(audio_path: str, request_id="unknown") -> FeatureResult:
    """
    提取7个目标特征（严格按照原始代码逻辑）
    特征列表：
    1. F0_max_std - 各语音段F0_max的标准差
    2. Delta0_mean - 各语音段Delta0的均值
    3. Delta2_mean - 各语音段Delta2的均值
    4. F0_slope_mean - 各语音段F0_slope的均值
    5. F2_std_mean - 各语音段F2_std的均值
    6. DFA_mean - 各语音段DFA的均值
    7. MFCC4_mean - 各语音段MFCC4的均值
    """
    warnings = []
    segment_features = []
    
    try:
        logger.info(f"请求[{request_id}] 开始特征提取: {audio_path}")
        
        # 加载音频
        logger.debug(f"请求[{request_id}] 加载音频...")
        y_raw, _ = librosa.load(audio_path, sr=CONFIG["SAMPLING_RATE"], mono=True)
        logger.debug(f"请求[{request_id}] 音频原始长度: {len(y_raw)} 采样点, 时长: {len(y_raw)/CONFIG['SAMPLING_RATE']:.2f}秒")
        
        # 静音修剪
        y_trimmed, _ = librosa.effects.trim(y_raw, top_db=20)
        logger.debug(f"请求[{request_id}] 修剪后长度: {len(y_trimmed)} 采样点")
        
        # 取前0.3秒作为噪音样本
        noise_sample = y_trimmed[:int(0.3 * CONFIG["SAMPLING_RATE"])]
        
        # 降噪
        logger.debug(f"请求[{request_id}] 开始降噪...")
        y_denoised = reduce_noise(
            y=y_trimmed,
            y_noise=noise_sample,
            sr=CONFIG["SAMPLING_RATE"],
            stationary=True,
            prop_decrease=0.6
        )
        logger.debug(f"请求[{request_id}] 降噪完成")
        
        # 分割语音段
        intervals = librosa.effects.split(
            y_denoised,
            top_db=25,
            frame_length=512,
            hop_length=128
        )
        
        logger.info(f"请求[{request_id}] 检测到 {len(intervals)} 个语音段")
        
        if len(intervals) == 0:
            logger.warning(f"请求[{request_id}] 未检测到语音段，使用整段音频")
            intervals = [(0, len(y_denoised))]
        
        # 处理每个语音段
        for idx, inter in enumerate(intervals):
            start, end = inter[0], inter[1]
            dur = (end - start) / CONFIG["SAMPLING_RATE"]
            
            logger.debug(f"请求[{request_id}] 语音段{idx}: 起点={start}, 终点={end}, 时长={dur:.2f}秒")
            
            if dur < 0.3:
                logger.debug(f"请求[{request_id}] 语音段{idx} 太短，跳过")
                continue
            
            # 提取原始音频段用于Praat特征
            seg_raw = y_trimmed[start:end].astype(np.float32)
            # 提取降噪后的音频段用于MFCC特征
            seg_denoised = y_denoised[start:end].astype(np.float32)
            
            # 提取每个语音段的原始特征（注意：这里提取的是基础特征，不是最终的_mean/_std）
            f0_max = extract_f0_max(seg_raw, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=idx)
            f0_slope = extract_f0_slope(seg_raw, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=idx)
            f2_std = extract_f2_std(seg_raw, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=idx)
            dfa = extract_dfa(seg_raw, request_id=request_id, segment_idx=idx)
            mfcc4, delta0, delta2 = extract_mfcc_and_delta(seg_denoised, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=idx)
            
            # 保存每个语音段的基础特征
            segment_features.append({
                'F0_max': f0_max,
                'F0_slope': f0_slope,
                'F2_std': f2_std,
                'DFA': dfa,
                'MFCC4': mfcc4,
                'Delta0': delta0,
                'Delta2': delta2
            })
        
        # 如果没有有效语音段，使用整段
        if not segment_features:
            logger.warning(f"请求[{request_id}] 没有有效语音段，使用整段音频")
            seg_raw = y_trimmed.astype(np.float32)
            seg_denoised = y_denoised.astype(np.float32)
            
            f0_max = extract_f0_max(seg_raw, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=0)
            f0_slope = extract_f0_slope(seg_raw, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=0)
            f2_std = extract_f2_std(seg_raw, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=0)
            dfa = extract_dfa(seg_raw, request_id=request_id, segment_idx=0)
            mfcc4, delta0, delta2 = extract_mfcc_and_delta(seg_denoised, CONFIG["SAMPLING_RATE"], request_id=request_id, segment_idx=0)
            
            segment_features.append({
                'F0_max': f0_max,
                'F0_slope': f0_slope,
                'F2_std': f2_std,
                'DFA': dfa,
                'MFCC4': mfcc4,
                'Delta0': delta0,
                'Delta2': delta2
            })
        
        # 转换为DataFrame进行聚合
        df = pd.DataFrame(segment_features)
        logger.info(f"请求[{request_id}] 有效语音段数: {len(df)}")
        
        # 关键修改：按照原始代码的逻辑计算特征
        # 1. 对MFCC和Delta特征取均值（它们已经在每个语音段内是均值了）
        # 2. 对其他特征（非MFCC/Delta）可以取均值和标准差，但这里我们只需要特定的7个
        
        # 计算最终需要的7个特征
        feature_dict = {
            # F0_max_std：各语音段F0_max的标准差
            "F0_max_std": float(df['F0_max'].std()) if len(df) > 1 else 0.0,
            
            # Delta0_mean：各语音段Delta0的均值
            "Delta0_mean": float(df['Delta0'].mean()),
            
            # Delta2_mean：各语音段Delta2的均值
            "Delta2_mean": float(df['Delta2'].mean()),
            
            # F0_slope_mean：各语音段F0_slope的均值
            "F0_slope_mean": float(df['F0_slope'].mean()),
            
            # F2_std_mean：各语音段F2_std的均值
            "F2_std_mean": float(df['F2_std'].mean()),
            
            # DFA_mean：各语音段DFA的均值
            "DFA_mean": float(df['DFA'].mean()),
            
            # MFCC4_mean：各语音段MFCC4的均值
            "MFCC4_mean": float(df['MFCC4'].mean())
        }
        
        # 记录每个特征的统计信息
        logger.info(f"请求[{request_id}] 特征统计:")
        for feat_name, feat_value in feature_dict.items():
            logger.info(f"  {feat_name}: {feat_value:.4f}")
        
        # 记录分段特征（用于调试）
        for idx, seg_feat in enumerate(segment_features):
            logger.debug(f"请求[{request_id}] 分段{idx}特征: {seg_feat}")
        
        del y_raw, y_trimmed, y_denoised, df
        gc.collect()

        return FeatureResult(
            features=np.array([feature_dict[name] for name in CONFIG["TARGET_FEATURES"]]),
            feature_dict=feature_dict,
            feature_warnings=warnings,
            raw_segment_features=[]  # 👈 清空
        )
        
    except Exception as e:
        logger.error(f"请求[{request_id}] 特征提取失败: {e}", exc_info=True)
        zero_features = {name: 0.0 for name in CONFIG["TARGET_FEATURES"]}
        return FeatureResult(
            features=np.zeros(len(CONFIG["TARGET_FEATURES"])),
            feature_dict=zero_features,
            feature_warnings=[f"严重错误：语音特征提取失败: {str(e)}"],
            raw_segment_features=[]
        )

def extract_7features_with_cache(audio_path: str, request_id="unknown") -> FeatureResult:
    """带缓存的特征提取"""
    cache_key = FeatureCache.get_key(audio_path)
    cached_result = FeatureCache.get(cache_key)
    if cached_result:
        logger.info(f"请求[{request_id}] 缓存命中")
        return cached_result
    
    logger.info(f"请求[{request_id}] 缓存未命中，开始特征提取")
    feature_result = extract_7features(audio_path, request_id)
    
    if not feature_result.feature_warnings or "失败" not in feature_result.feature_warnings[0]:
        FeatureCache.set(cache_key, feature_result)
        logger.info(f"请求[{request_id}] 特征结果已缓存")
    
    return feature_result

# ===================== PD亚型概率计算 =====================
def calculate_subtype_probs(feature_dict: Dict[str, float], pd_prob: float, request_id="unknown") -> Dict[str, float]:
    """计算PD亚型概率"""
    weights = CONFIG["SUBTYPE_WEIGHTS"]
    subtype_scores = {}
    
    logger.debug(f"请求[{request_id}] 开始计算亚型概率")
    
    for subtype, feat_weights in weights.items():
        score = 0.0
        for feat_name, weight in feat_weights.items():
            feat_value = feature_dict.get(feat_name, 0.0)
            score += feat_value * weight
            logger.debug(f"请求[{request_id}] {subtype} - {feat_name}: {feat_value:.4f} * {weight} = {feat_value * weight:.4f}")
        subtype_scores[subtype] = score
        logger.debug(f"请求[{request_id}] {subtype} 原始得分: {score:.4f}")
    
    adjusted_scores = {}
    for subtype, score in subtype_scores.items():
        adjusted_scores[subtype] = score * pd_prob * CONFIG["SUBTYPE_SCALE"] + CONFIG["SUBTYPE_SMOOTH"]
        logger.debug(f"请求[{request_id}] {subtype} 调整后得分: {adjusted_scores[subtype]:.4f}")
    
    sigmoid_scores = {subtype: sigmoid(score) for subtype, score in adjusted_scores.items()}
    logger.debug(f"请求[{request_id}] Sigmoid后得分: {sigmoid_scores}")
    
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
    
    logger.info(f"请求[{request_id}] 亚型概率计算完成: {normalized_probs}")
    return normalized_probs

# ===================== 诊断逻辑 =====================
def diagnose(feature_result: FeatureResult, request_id="unknown") -> DiagnosisResult:
    """使用模型诊断"""
    model, preprocessor, lasso_features = ModelManager.load_all()
    
    start_time = time.time()
    
    # 特征预处理
    logger.info(f"请求[{request_id}] 开始特征预处理")
    features_scaled = ModelManager.preprocess_features(feature_result.feature_dict, request_id)
    
    # 预测
    try:
        if hasattr(model, 'predict_proba'):
            logger.debug(f"请求[{request_id}] 使用predict_proba预测")
            proba = model.predict_proba(features_scaled)[0]
            pd_prob = float(proba[1])
            logger.info(f"请求[{request_id}] 预测概率分布: [健康={proba[0]:.4f}, PD={proba[1]:.4f}]")
        else:
            logger.debug(f"请求[{request_id}] 使用predict预测")
            pred = model.predict(features_scaled)[0]
            pd_prob = 0.9 if pred == 1 else 0.1
            logger.info(f"请求[{request_id}] 预测类别: {pred}")
            
        logger.info(f"请求[{request_id}] 预测成功: pd_prob={pd_prob:.4f}")
    except Exception as e:
        logger.error(f"请求[{request_id}] 预测失败: {e}", exc_info=True)
        pd_prob = 0.5
    
    # 计算亚型概率
    subtype_probs = calculate_subtype_probs(feature_result.feature_dict, pd_prob, request_id)
    
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
        # raw_features_scaled 整行删除
    )

# ===================== Flask API =====================
app = Flask(__name__)
CORS(app)

@app.before_request
def before_request():
    """请求前处理"""
    g.start_time = time.time()
    g.request_id = hashlib.md5(f"{time.time()}_{np.random.random()}".encode()).hexdigest()[:8]
    logger.info(f"请求[{g.request_id}]开始: {request.path} 方法: {request.method}")

@app.after_request
def after_request(response):
    """请求后处理"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"请求[{getattr(g, 'request_id', 'unknown')}]完成: 耗时{duration:.2f}秒 状态码: {response.status_code}")
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
        logger.info(f"请求[{request_id}] 收到诊断请求")
        
        if 'file' not in request.files:
            logger.warning(f"请求[{request_id}] 未上传音频文件")
            return jsonify({"code": 400, "msg": "未上传音频文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning(f"请求[{request_id}] 未选择音频文件")
            return jsonify({"code": 400, "msg": "未选择音频文件"}), 400
        
        if not check_ffmpeg():
            logger.error(f"请求[{request_id}] FFmpeg不可用")
            return jsonify({"code": 500, "msg": "服务器音频处理组件缺失"}), 500
        
        logger.info(f"请求[{request_id}] 接收文件: {file.filename}, 大小: {file.content_length if file.content_length else '未知'}字节")
        
        temp_path = get_temp_path("upload")
        converted_path = get_temp_path("converted")
        temp_paths = [temp_path, converted_path]
        
        file.save(temp_path)
        logger.info(f"请求[{request_id}] 临时文件已保存: {temp_path}")
        
        # 音频转换
        logger.info(f"请求[{request_id}] 开始音频转换")
        try:
            convert_result = convert_audio_sync(temp_path, converted_path, request_id)
            if not convert_result:
                raise Exception("音频转换失败")
            logger.info(f"请求[{request_id}] 音频转换完成: {converted_path}")
        except Exception as e:
            logger.error(f"请求[{request_id}] 音频转换异常: {e}", exc_info=True)
            return jsonify({"code": 500, "msg": f"音频转换失败: {str(e)}"}), 500
        
        # 验证时长
        valid, duration, msg = validate_audio_duration(converted_path)
        if not valid:
            logger.warning(f"请求[{request_id}] 音频时长验证失败: {msg}")
            return jsonify({"code": 400, "msg": msg}), 400
        
        logger.info(f"请求[{request_id}] 音频时长: {duration:.2f}秒")
        
        # 特征提取
        logger.info(f"请求[{request_id}] 开始特征提取")
        try:
            with ThreadPoolExecutor(max_workers=1) as thread_executor:
                future_feat = thread_executor.submit(extract_7features_with_cache, converted_path, request_id)
                feature_result = future_feat.result(timeout=CONFIG["ANALYSIS_TIMEOUT_SECONDS"])
            logger.info(f"请求[{request_id}] 特征提取完成")
        except TimeoutError:
            logger.error(f"请求[{request_id}] 特征提取超时")
            return jsonify({
                "code": 504,
                "msg": "特征提取超时，请稍后重试"
            }), 504
        except Exception as e:
            logger.error(f"请求[{request_id}] 特征提取失败: {e}", exc_info=True)
            return jsonify({"code": 500, "msg": f"特征提取失败: {str(e)}"}), 500
        
        # 检查特征提取是否成功
        if feature_result.feature_warnings and "失败" in feature_result.feature_warnings[0]:
            logger.error(f"请求[{request_id}] 特征提取错误: {feature_result.feature_warnings[0]}")
            return jsonify({
                "code": 500,
                "msg": feature_result.feature_warnings[0]
            }), 500
        
        # 诊断
        logger.info(f"请求[{request_id}] 开始诊断")
        try:
            with ThreadPoolExecutor(max_workers=1) as thread_executor:
                future_diag = thread_executor.submit(diagnose, feature_result, request_id)
                diagnosis_result = future_diag.result(timeout=30)
            logger.info(f"请求[{request_id}] 诊断完成")
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
        
        logger.info(f"请求[{request_id}] 诊断完成，总耗时{total_time:.2f}秒")
        
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

# ===================== 调试接口 =====================
@app.route('/debug/feature/<path:filename>', methods=['GET'])
def debug_feature(filename):
    """调试接口：直接提取指定音频文件的特征"""
    try:
        filepath = os.path.join(CONFIG["TEMP_DIR"], filename)
        if not os.path.exists(filepath):
            return jsonify({"code": 404, "msg": "文件不存在"}), 404
        
        request_id = f"debug_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
        feature_result = extract_7features(filepath, request_id)
        
        return jsonify({
            "code": 200,
            "data": {
                "features": feature_result.feature_dict,
                "warnings": feature_result.feature_warnings,
                "segment_count": len(feature_result.raw_segment_features) if feature_result.raw_segment_features else 0
            }
        }), 200
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500

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
    print("="*60)
    print("PD语音诊断服务启动中...")
    print("="*60)
    
    if not check_ffmpeg():
        logger.error("ffmpeg未安装，请先安装ffmpeg")
        exit(1)
    
    try:
        ModelManager.load_all()
        logger.info("✅ 模型加载成功")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        exit(1)
    
    cleanup_temp_files(glob.glob(os.path.join(CONFIG["TEMP_DIR"], "*.wav")))
    
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"🚀 服务启动在端口 {port}，调试模式: {debug}")
    logger.info(f"📝 详细日志将写入: pd_diagnoser_detailed.log")
    logger.info(f"⚠️  错误日志将写入: pd_diagnoser_errors.log")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
