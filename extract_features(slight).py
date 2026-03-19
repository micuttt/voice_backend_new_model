import librosa
import numpy as np
import pandas as pd
import os
from noisereduce import reduce_noise
from scipy.stats import linregress
import antropy as ant
import warnings

import parselmouth
from parselmouth.praat import call
import concurrent.futures
from tqdm import tqdm

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. 帕金森诊断标准：Praat 黄金特征（已移除 CPP）
# -----------------------------------------------------------------------------
def get_praat_features(seg, sr, fmin=75, fmax=500):
    feats = {
        'F0_mean': 0, 'F0_std': 0, 'F0_max': 0, 'F0_min': 0, 'F0_slope': 0,
        'Jitter_rel': 0,
        'Shim_loc': 0,
        'HNR': 0, 'NHR': 0,
        'F1_mean':0,'F2_mean':0,'F1_std':0,'F2_std':0
    }

    try:
        snd = parselmouth.Sound(seg, sr)
        pitch = call(snd, "To Pitch", 0.001, fmin, fmax)
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")

        formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1_mean = call(formant, "Get mean", 1, 0, 0, "Hertz")
        f2_mean = call(formant, "Get mean", 2, 0, 0, "Hertz")
        f1_std = call(formant, "Get standard deviation", 1, 0, 0, "Hertz")
        f2_std = call(formant, "Get standard deviation", 2, 0, 0, "Hertz")

        if not np.isnan(f0_mean):
            feats['F0_mean'] = f0_mean
            feats['F0_std'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            feats['F0_max'] = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            feats['F0_min'] = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            
            pitch_arr = pitch.selected_array['frequency']
            pitch_arr = pitch_arr[pitch_arr > 0]
            if len(pitch_arr) > 2:
                feats['F0_slope'] = linregress(range(len(pitch_arr)), pitch_arr)[0]

        if not np.isnan(f1_mean):
            feats['F1_mean'] = f1_mean
            feats['F2_mean'] = f2_mean
            feats['F1_std'] = f1_std   
            feats['F2_std'] = f2_std

        pointProcess = call(snd, "To PointProcess (periodic, cc)", fmin, fmax)
        num_periods = call(pointProcess, "Get number of periods", 0, 0, 1e-4, 0.02, 1.3)
        
        if num_periods > 8:
            feats['Jitter_rel'] = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            feats['Shim_loc']   = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        if not np.isnan(hnr) and hnr > 0:
            feats['HNR'] = hnr
            feats['NHR'] = 10**(-hnr/20)

    except Exception:
        pass

    for k in feats:
        if np.isnan(feats[k]): feats[k] = 0
    return feats

# -----------------------------------------------------------------------------
# 2. 宏观频谱特征（已修复 rolloff）
# -----------------------------------------------------------------------------
def get_voice_features(seg, sr):
    feats = {}
    spec_flat = librosa.feature.spectral_flatness(y=seg).mean()
    feats['Inverse_Spec_Flatness'] = 1.0 - spec_flat
    rms = librosa.feature.rms(y=seg)[0]
    feats['RMS_energy'] = rms.mean()
    feats['Energy_std'] = rms.std()
    feats['Spec_cent'] = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
    feats['Spec_bandwidth'] = librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean()
    feats['Spec_rolloff'] = librosa.feature.spectral_rolloff(y=seg, sr=sr).mean()
    feats['Zero_cross_rate'] = librosa.feature.zero_crossing_rate(seg).mean()
    return feats

# -----------------------------------------------------------------------------
# 3. 非线性动力学特征
# -----------------------------------------------------------------------------
def get_nonlinear_feats(seg, sr=16000):
    feats = {}
    try:
        if len(seg) < 200:
            return {k: 0 for k in['LZC', 'DFA', 'PermEn', 'SampleEntropy']}
            
        bin_sig = seg > np.mean(seg)
        feats['LZC'] = ant.lziv_complexity(bin_sig, normalize=True)
        feats['DFA'] = ant.detrended_fluctuation(seg)
        feats['PermEn'] = ant.perm_entropy(seg, order=3, delay=1)
        
        if len(seg) > 2000:
            seg_ds = librosa.resample(seg, orig_sr=sr, target_sr=4000)
            feats['SampleEntropy'] = ant.sample_entropy(seg_ds)
        else:
            feats['SampleEntropy'] = ant.sample_entropy(seg)
    except:
        feats = {'LZC': 0, 'DFA': 0, 'PermEn': 0, 'SampleEntropy': 0}
    return feats

# -----------------------------------------------------------------------------
# 4. MFCC 特征
# -----------------------------------------------------------------------------
def get_mfcc_features(seg, sr, n_mfcc=6):
    feats = {}
    try:
        mfccs = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=128)
        delta = librosa.feature.delta(mfccs)
        for i in range(n_mfcc):
            feats[f'MFCC{i}'] = mfccs[i].mean()
            feats[f'Delta{i}'] = delta[i].mean()
    except:
        pass
    return feats

# -----------------------------------------------------------------------------
# 5. 核心引擎（已移除 Duration）
# -----------------------------------------------------------------------------
def process_one_audio(audio_path, sr=16000, min_dur=0.3):
    try:
        y_raw, _ = librosa.load(audio_path, sr=sr, mono=True)
        y_raw_trim, _ = librosa.effects.trim(y_raw, top_db=20)
        
        noise_sample = y_raw_trim[:int(0.3 * sr)]  # 取前0.3秒当噪音
        y_denoised = reduce_noise(
            y=y_raw_trim,
            y_noise=noise_sample,
            sr=sr,
            stationary=True,
            prop_decrease=0.6
        )
        intervals = librosa.effects.split(y_denoised, top_db=25, frame_length=512, hop_length=128)
        
        segs_features =[]
        
        for inter in intervals:
            start, end = inter[0], inter[1]
            dur = (end - start) / sr
            
            if dur < min_dur: continue
                
            seg_raw = y_raw_trim[start:end].astype(np.float32)
            seg_denoised = y_denoised[start:end].astype(np.float32)
            
            feats = {}
            feats.update(get_praat_features(seg_raw, sr))
            feats.update(get_nonlinear_feats(seg_raw, sr))
            feats.update(get_voice_features(seg_denoised, sr))
            feats.update(get_mfcc_features(seg_denoised, sr)) 
            
            segs_features.append(feats)
            
        if not segs_features:
            return None

        df = pd.DataFrame(segs_features)
        mean_feats = df.mean().add_suffix('_mean')
        non_mfcc_cols =[col for col in df.columns if 'MFCC' not in col and 'Delta' not in col]
        std_feats = df[non_mfcc_cols].std().fillna(0).add_suffix('_std')
        
        final_feats = pd.concat([mean_feats, std_feats]).to_dict()
        
        return final_feats

    except Exception as e:
        print(f"⚠️  跳过 {os.path.basename(audio_path)}: {str(e)[:50]}")
        return None

# -----------------------------------------------------------------------------
# 6. 批量处理
# -----------------------------------------------------------------------------
def batch_all_subfolders(root_folder="."):
    for item in os.listdir(root_folder):
        sub = os.path.join(root_folder, item)
        if not os.path.isdir(sub) or item.startswith('.'): 
            continue
            
        wavs = sorted([f for f in os.listdir(sub) if f.lower().endswith('.wav')])
        if not wavs: 
            continue
            
        print(f"\n📂 正在处理文件夹：{item} | 包含 {len(wavs)} 个音频文件")

        tasks =[(os.path.join(sub, w), w, i) for i, w in enumerate(wavs)]
        res_list =[]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for res in tqdm(executor.map(process_file_wrapper, tasks), total=len(tasks)):
                if res:
                    res_list.append(res)

        if res_list:
            df_out = pd.DataFrame(res_list)
            cols = ['idx', 'file'] +[c for c in df_out.columns if c not in ['idx', 'file']]
            df_out = df_out[cols]
            
            csv_name = f"{item}_parkinson_features(slight).csv"
            df_out.to_csv(csv_name, index=False, encoding='utf-8-sig')
            print(f"✅ 完成提取！特征库已保存为：{csv_name}")

def process_file_wrapper(args):
    file_path, w, i = args
    res = process_one_audio(file_path)
    if res:
        res['file'] = w
        res['idx'] = i + 1
    return res

if __name__ == "__main__":
    print("🚀 正在初始化帕金森语音医学特征提取引擎...")
    batch_all_subfolders()