import os
import glob
import logging
import numpy as np
import librosa
import parselmouth
from flask import Flask, request, jsonify
from flask_cors import CORS
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
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
from functools import wraps

# ===================== 核心配置 =====================
CONFIG = {
    "SAMPLING_RATE": 16000,
    "TEMP_DIR": tempfile.gettempdir(),
    "MODEL_PATH": "./model_rank1_Bernoulli_NB.pkl",
    "PREPROCESSOR_PATH": "./preprocessor.pkl",
    "FEATURE_ORDER_PATH": "./shap_importance.pkl",  # 存储LASSO筛选后的特征顺序
    "PD_THRESHOLD": 0.5,
    "MIN_AUDIO_DURATION": 1.0,
    "MAX_AUDIO_DURATION": 210.0,
    
    # 7个目标特征（从SHAP重要性文件得知）
    "TARGET_FEATURES": [
        "Delta2_mean",
        "DFA_mean", 
        "F2_std_mean",
        "F0_slope_mean",
        "Delta0_mean",
        "MFCC3_mean",
        "F2_std_std"
    ],
    
    # 亚型分类配置（基于7个特征的权重）
    "SUBTYPE_WEIGHTS": {
        "tremor": {
            "Delta2_mean": 0.25,    # 高频变化，与震颤相关
            "DFA_mean": 0.20,        # 波动复杂度
            "F0_slope_mean": 0.15,   # 基频变化
            "Delta0_mean": 0.25,      # 低频变化
            "MFCC3_mean": 0.15        # 声道配置
        },
        "rigidity": {
            "F0_slope_mean": -0.30,   # 僵直导致基频变化减少
            "F2_std_mean": -0.25,     # 第二共振峰变化减少
            "F2_std_std": -0.20,      # 共振峰稳定性降低
            "Delta2_mean": -0.15,      # 高频变化减少
            "MFCC3_mean": -0.10        # 声道灵活性降低
        },
        "motor": {
            "DFA_mean": 0.25,         # 复杂运动控制
            "Delta2_mean": 0.20,       # 高频运动成分
            "F0_slope_mean": 0.20,     # 基频变化
            "F2_std_mean": 0.20,       # 共振峰变化
            "F2_std_std": 0.15         # 共振峰稳定性
        },
        "non_motor": {
            "MFCC3_mean": 0.30,        # 声道配置，与非运动症状相关
            "Delta0_mean": 0.25,        # 缓慢变化
            "DFA_mean": 0.25,           # 波动特性
            "F0_slope_mean": 0.20       # 基频变化
        }
    },
    
    # 亚型概率归一化参数
    "SUBTYPE_SMOOTH": 0.1,
    "SUBTYPE_SCALE": 1.2,
    
    # 亚型解释说明
    "SUBTYPE_EXPLANATION": {
        "tremor": "震颤主导型：以Delta2_mean（高频变化）和Delta0_mean（低频变化）升高为特征，反映语音震颤",
        "rigidity": "僵直主导型：以F0_slope_mean（基频斜率）和F2_std（共振峰变化）降低为特征，反映发声器官僵直",
        "motor": "运动型：以DFA_mean（波动复杂度）和Delta2_mean（高频变化）升高为特征，反映运动症状影响",
        "non_motor": "非运动型：以MFCC3_mean（声道配置）和Delta0_mean（缓慢变化）改变为特征，反映非运动症状影响"
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

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PDDiagnoser")

# ===================== 数据类 =====================
@dataclass
class FeatureResult:
    features: np.ndarray  # 7个特征向量
    feature_dict: Dict[str, float]  # 特征字典
    feature_warnings: List[str]

@dataclass
class DiagnosisResult:
    pd_prob: float
    diagnosis: str
    risk: str
    features: Dict[str, float]
    subtype_probs: Dict[str, float]  # 新增：亚型概率
    feature_warnings: List[str]
    processing_time: float

# ===================== 限流装饰器 =====================
class RateLimiter:
    def __init__(self):
        self.requests = {}
        
    def is_allowed(self, ip: str) -> Tuple[bool, str]:
        import time
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

# ===================== 模型加载 =====================
class ModelManager:
    _model = None
    _preprocessor = None
    _lasso_features = None  # LASSO筛选后的特征（7个）
    _subtype_classifier = None  # 新增：亚型分类器

    @classmethod
    def load_all(cls):
        """加载模型、预处理器和LASSO筛选后的特征"""
        
        # 1. 加载LASSO筛选后的特征顺序
        if cls._lasso_features is None:
            try:
                # 从shap_importance.pkl读取特征顺序
                feature_data = joblib.load(CONFIG["FEATURE_ORDER_PATH"])
                if isinstance(feature_data, pd.DataFrame):
                    cls._lasso_features = feature_data['feature'].tolist()
                elif isinstance(feature_data, list):
                    cls._lasso_features = feature_data
                else:
                    cls._lasso_features = CONFIG["TARGET_FEATURES"]
                logger.info(f"加载LASSO筛选后的特征成功，共 {len(cls._lasso_features)} 个: {cls._lasso_features}")
            except Exception as e:
                logger.error(f"加载特征顺序失败: {e}")
                cls._lasso_features = CONFIG["TARGET_FEATURES"]
                logger.info(f"使用默认特征顺序")

        # 2. 加载预处理器（用于标准化）
        if cls._preprocessor is None:
            try:
                cls._preprocessor = joblib.load(CONFIG["PREPROCESSOR_PATH"])
                logger.info(f"加载预处理器成功")
                
                # 获取所有特征信息（仅用于标准化）
                small_cols = cls._preprocessor.get('small_cols', [])
                large_cols = cls._preprocessor.get('large_cols', [])
                logger.info(f"预处理器包含小范围特征: {len(small_cols)}个，大范围特征: {len(large_cols)}个")
                
            except Exception as e:
                logger.error(f"加载预处理器失败: {e}")
                raise

        # 3. 加载模型
        if cls._model is None:
            try:
                cls._model = joblib.load(CONFIG["MODEL_PATH"])
                logger.info(f"加载模型成功: {type(cls._model)}")
                
                # 检查模型期望的特征数
                if hasattr(cls._model, 'n_features_in_'):
                    logger.info(f"模型期望特征数: {cls._model.n_features_in_}")
                    if cls._model.n_features_in_ == 7:
                        logger.info("✅ 模型期望7个特征，与LASSO筛选结果一致")
                    else:
                        logger.warning(f"⚠️ 模型期望 {cls._model.n_features_in_} 个特征")
                        
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                raise
        
        # 4. 尝试加载亚型分类器（可选）
        if cls._subtype_classifier is None:
            try:
                subtype_model_path = CONFIG.get("SUBTYPE_MODEL_PATH", "./subtype_classifier.pkl")
                cls._subtype_classifier = joblib.load(subtype_model_path)
                logger.info("加载亚型预训练分类器成功")
            except (FileNotFoundError, KeyError):
                logger.info("未找到预训练亚型分类器，使用基于文献的规则分类法")
                cls._subtype_classifier = "rule_based"
            except Exception as e:
                logger.warning(f"加载亚型分类器失败: {e}，使用规则分类法")
                cls._subtype_classifier = "rule_based"

        return cls._model, cls._preprocessor, cls._lasso_features

    @classmethod
    def preprocess_features(cls, features_dict):
        """
        使用训练时的预处理器对7个特征进行标准化
        返回：标准化后的7个特征向量（按LASSO顺序）
        """
        if cls._preprocessor is None:
            cls.load_all()
        
        # 获取预处理器
        small_scaler = cls._preprocessor['small']
        large_scaler = cls._preprocessor['large']
        small_cols = cls._preprocessor['small_cols']
        large_cols = cls._preprocessor['large_cols']
        
        # 创建包含所有特征的DataFrame用于标准化
        all_features = small_cols + large_cols
        all_features_df = pd.DataFrame(0.0, index=[0], columns=all_features)
        
        # 填充7个特征
        for feat_name, value in features_dict.items():
            if feat_name in all_features_df.columns:
                all_features_df[feat_name] = value
        
        # 分别标准化
        features_scaled = all_features_df.copy()
        
        if small_cols:
            existing_small = [c for c in small_cols if c in features_scaled.columns]
            if existing_small:
                features_scaled[existing_small] = small_scaler.transform(features_scaled[existing_small])
                
        if large_cols:
            existing_large = [c for c in large_cols if c in features_scaled.columns]
            if existing_large:
                features_scaled[existing_large] = large_scaler.transform(features_scaled[existing_large])
        
        # 只提取LASSO筛选后的7个特征（标准化后的值）
        lasso_features_scaled = []
        for feat_name in cls._lasso_features:
            if feat_name in features_scaled.columns:
                lasso_features_scaled.append(features_scaled[feat_name].iloc[0])
            else:
                logger.warning(f"特征 {feat_name} 不在预处理器中，用0填充")
                lasso_features_scaled.append(0.0)
        
        return np.array(lasso_features_scaled).reshape(1, -1)

# ===================== 工具函数 =====================
def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, 
                               text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def validate_audio_duration(audio_path: str) -> Tuple[bool, float, str]:
    """验证音频时长是否在允许范围内"""
    try:
        import wave
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate
            
        if duration < CONFIG["MIN_AUDIO_DURATION"]:
            return False, duration, f"音频过短 ({duration:.2f}秒)，需大于{CONFIG['MIN_AUDIO_DURATION']}秒"
        
        if duration > CONFIG["MAX_AUDIO_DURATION"]:
            return False, duration, f"音频过长 ({duration:.2f}秒)，需小于{CONFIG['MAX_AUDIO_DURATION']}秒"
        
        return True, duration, ""
    except Exception as e:
        return False, 0, f"读取音频失败: {str(e)}"

def clean_temp_files(pattern: str = "*.wav"):
    try:
        for file in glob.glob(os.path.join(CONFIG["TEMP_DIR"], pattern)):
            os.remove(file)
    except Exception as e:
        logger.warning(f"清理临时文件失败: {e}")

def get_temp_path(prefix: str) -> str:
    return os.path.join(CONFIG["TEMP_DIR"], f"{prefix}_{os.getpid()}_{np.random.randint(10000)}.wav")

def convert_numpy_type(obj):
    """递归转换numpy类型为Python原生类型"""
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
            feature_warnings=[f"特征提取失败: {str(e)}"]
        )

# ===================== PD亚型概率计算（新增） =====================
def calculate_subtype_probs(feature_dict: Dict[str, float], pd_prob: float) -> Dict[str, float]:
    """
    基于7个特征计算PD亚型概率
    完全模仿第一个代码的亚型计算逻辑
    """
    weights = CONFIG["SUBTYPE_WEIGHTS"]
    subtype_scores = {}
    
    # 1. 计算各亚型原始得分（特征加权和）
    for subtype, feat_weights in weights.items():
        score = 0.0
        for feat_name, weight in feat_weights.items():
            # 获取特征值，如果不存在则用0
            feat_value = feature_dict.get(feat_name, 0.0)
            score += feat_value * weight
        subtype_scores[subtype] = score
    
    # 2. 结合PD概率调整得分（模仿第一个代码）
    adjusted_scores = {}
    for subtype, score in subtype_scores.items():
        adjusted_scores[subtype] = score * pd_prob * CONFIG["SUBTYPE_SCALE"] + CONFIG["SUBTYPE_SMOOTH"]
    
    # 3. Sigmoid转换（将得分映射到0-1之间）
    sigmoid_scores = {subtype: sigmoid(score) for subtype, score in adjusted_scores.items()}
    
    # 4. 归一化（确保和为1）
    total = sum(sigmoid_scores.values())
    if total > 0:
        normalized_probs = {
            subtype: round(prob / total, 4) 
            for subtype, prob in sigmoid_scores.items()
        }
    else:
        # 如果总和为0，则平均分配
        n_subtypes = len(sigmoid_scores)
        normalized_probs = {subtype: round(1.0 / n_subtypes, 4) for subtype in sigmoid_scores.keys()}
    
    # 5. 低PD概率时降低亚型概率（模仿第一个代码）
    if pd_prob < CONFIG["PD_THRESHOLD"]:
        for subtype in normalized_probs:
            normalized_probs[subtype] = round(normalized_probs[subtype] * pd_prob, 4)
    
    logger.info(f"亚型概率计算完成: {normalized_probs}")
    return normalized_probs

# ===================== 诊断逻辑 =====================
def diagnose(feature_result: FeatureResult) -> DiagnosisResult:
    """使用模型诊断，并计算亚型概率"""
    model, preprocessor, lasso_features = ModelManager.load_all()
    
    start_time = time.time()
    
    # 1. 使用预处理器标准化7个特征
    features_scaled = ModelManager.preprocess_features(feature_result.feature_dict)
    
    # 2. 预测（features_scaled已经是7个特征）
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
    
    # 3. 计算亚型概率（新增）
    subtype_probs = calculate_subtype_probs(feature_result.feature_dict, pd_prob)
    
    # 4. 诊断结论
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
        subtype_probs=subtype_probs,  # 新增
        feature_warnings=feature_result.feature_warnings,
        processing_time=round(processing_time, 3)
    )

# ===================== Flask API =====================
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health_check():
    return "PD Voice Diagnosis Service", 200

@app.route('/analyze', methods=['POST'])
@rate_limit
def pd_diagnose():
    temp_paths = []
    total_start = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({"code": 400, "msg": "未上传音频文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"code": 400, "msg": "未选择音频文件"}), 400
        
        if not check_ffmpeg():
            return jsonify({"code": 500, "msg": "服务器音频处理组件缺失"}), 500
        
        logger.info(f"接收请求: {file.filename}")
        
        temp_path = get_temp_path("upload")
        converted_path = get_temp_path("converted")
        temp_paths = [temp_path, converted_path]
        
        file.save(temp_path)
        
        ffmpeg_cmd = [
            "ffmpeg", "-i", temp_path,
            "-ar", str(CONFIG["SAMPLING_RATE"]),
            "-ac", "1", "-sample_fmt", "s16",
            "-c:a", "pcm_s16le", "-y", converted_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"音频转换失败")
        
        valid, duration, msg = validate_audio_duration(converted_path)
        if not valid:
            return jsonify({"code": 400, "msg": msg}), 400
        
        logger.info("开始提取7个目标特征...")
        feature_result = extract_7features(converted_path)
        
        logger.info("开始诊断并计算亚型概率...")
        diagnosis_result = diagnose(feature_result)
        
        total_time = time.time() - total_start
        
        # 构建返回结果（包含亚型信息）
        response = {
            "code": 200,
            "msg": "诊断成功",
            "data": {
                "audio_duration": round(duration, 2),
                "processing_time": round(total_time, 2),
                "pd_probability": diagnosis_result.pd_prob,
                "diagnosis": diagnosis_result.diagnosis,
                "risk_level": diagnosis_result.risk,
                "features": diagnosis_result.features,
                "feature_warnings": diagnosis_result.feature_warnings,
                # 新增亚型相关信息
                "subtype_probabilities": {
                    "tremor_type": diagnosis_result.subtype_probs.get("tremor", 0.0),
                    "rigidity_type": diagnosis_result.subtype_probs.get("rigidity", 0.0),
                    "motor_type": diagnosis_result.subtype_probs.get("motor", 0.0),
                    "non_motor_type": diagnosis_result.subtype_probs.get("non_motor", 0.0)
                },
                "subtype_explanation": CONFIG["SUBTYPE_EXPLANATION"]
            }
        }
        
        # 转换numpy类型
        response = convert_numpy_type(response)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"诊断异常: {e}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"诊断失败: {str(e)}"
        }), 500
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

# ===================== 启动服务 =====================
if __name__ == '__main__':
    if not check_ffmpeg():
        logger.error("ffmpeg未安装，请先安装ffmpeg")
        exit(1)
    
    try:
        ModelManager.load_all()
        logger.info("服务启动成功")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        exit(1)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)