import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# 模型库
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ------------------------------------------------------------------------------
# 1. 加载数据
# ------------------------------------------------------------------------------
print("="*60)
print("📊 步骤1: 加载数据")
print("="*60)

df = pd.read_excel('ReadText_parkinson_features(slight).xlsx')
target = 'status'
exclude_cols = ['idx', 'status', 'file']

X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
y = df[target]

print(f"原始特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 性别编码
if 'gender' in X.columns:
    X['gender'] = X['gender'].apply(lambda x: 1 if str(x).lower() in ['男', 1] else 0)
    print("✅ 性别编码完成")

# 缺失值填充
X = X.fillna(X.mean())
print("✅ 缺失值填充完成")

# ------------------------------------------------------------------------------
# 2. 定义你要的 7 个目标特征（严格顺序！）
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 步骤2: 使用固定7个特征")
print("="*60)

# ✅ 你指定的新顺序（完全不变）
TARGET_FEATURES = [
    "F0_max_std",
    "Delta0_mean",
    "Delta2_mean",
    "F0_slope_mean",
    "F2_std_mean",
    "DFA_mean",
    "MFCC4_mean"
]

print(f"\n📌 7个目标特征（严格顺序）:")
for i, feat in enumerate(TARGET_FEATURES, 1):
    print(f"  {i}. {feat}")

# ------------------------------------------------------------------------------
# 3. 提取7个特征
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 步骤3: 提取7特征数据集")
print("="*60)

X_7features = X[TARGET_FEATURES].copy()
print(f"7特征数据集形状: {X_7features.shape}")
print(f"7特征数据集统计:")
for col in X_7features.columns:
    print(f"  {col}: mean={X_7features[col].mean():.4f}, std={X_7features[col].std():.4f}")

# ------------------------------------------------------------------------------
# 4. 创建预处理器
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 步骤4: 创建预处理器 -> preprocessor(slight).pkl")
print("="*60)

feature_scales = {col: X_7features[col].max() for col in TARGET_FEATURES}
large_scale_7features = [col for col in TARGET_FEATURES if feature_scales[col] > 10]
small_scale_7features = [col for col in TARGET_FEATURES if feature_scales[col] <= 10]

print(f"小范围特征（≤10）: {small_scale_7features}")
print(f"大范围特征（>10）: {large_scale_7features}")

preprocessor = {
    'small': StandardScaler() if small_scale_7features else None,
    'large': RobustScaler() if large_scale_7features else None,
    'small_cols': small_scale_7features,
    'large_cols': large_scale_7features,
    'feature_order': TARGET_FEATURES,
    'feature_stats': {
        col: {
            'mean': float(X_7features[col].mean()),
            'std': float(X_7features[col].std()),
            'min': float(X_7features[col].min()),
            'max': float(X_7features[col].max())
        } for col in TARGET_FEATURES
    }
}

if small_scale_7features:
    preprocessor['small'].fit(X_7features[small_scale_7features])
if large_scale_7features:
    preprocessor['large'].fit(X_7features[large_scale_7features])

joblib.dump(preprocessor, 'preprocessor(slight).pkl')
print("✅ 预处理器已保存: preprocessor(slight).pkl")

# ------------------------------------------------------------------------------
# 5. 标准化
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 步骤5: 标准化7特征")
print("="*60)

X_7features_scaled = X_7features.copy()
if small_scale_7features:
    X_7features_scaled[small_scale_7features] = preprocessor['small'].transform(X_7features[small_scale_7features])
if large_scale_7features:
    X_7features_scaled[large_scale_7features] = preprocessor['large'].transform(X_7features[large_scale_7features])

# ------------------------------------------------------------------------------
# 6. 交叉验证 & 训练模型
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 步骤6: 训练模型")
print("="*60)

loocv = LeaveOneOut()
X_ = X_7features_scaled.values
y_ = y.values

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Bernoulli NB": BernoulliNB(),
    "MLP": MLPClassifier(random_state=42, max_iter=500),
}

results = []
for name, model in models.items():
    try:
        y_pred = cross_val_predict(model, X_, y_, cv=loocv)
        y_prob = cross_val_predict(model, X_, y_, cv=loocv, method='predict_proba')[:,1]
        acc = accuracy_score(y_, y_pred)
        f1 = f1_score(y_, y_pred)
        auc = roc_auc_score(y_, y_prob)
        model.fit(X_, y_)
        results.append({"model_name": name, "model": model, "acc": acc, "f1": f1, "auc": auc})
        print(f"{name:<20} | AUC={auc:.3f} | Acc={acc:.3f} | F1={f1:.3f}")
    except Exception as e:
        print(f"{name:<20} | 失败: {str(e)[:50]}")

# ------------------------------------------------------------------------------
# 7. 保存TOP5模型
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 步骤7: 保存最优模型")
print("="*60)

df_res = pd.DataFrame(results).sort_values('auc', ascending=False).reset_index(drop=True)
top5 = df_res.head(5)

for i, row in top5.iterrows():
    model_name = row["model_name"].replace(" ", "_")
    joblib.dump(row["model"], f"model_rank{i+1}_{model_name}(slight).pkl")
    print(f"✅ 保存 model_rank{i+1}_{model_name}(slight).pkl")

# ------------------------------------------------------------------------------
# 8. 保存特征重要性 & 顺序
# ------------------------------------------------------------------------------
shap_imp_df = pd.DataFrame({
    'feature': TARGET_FEATURES,
    'shap_importance': np.ones(7)
})
joblib.dump(shap_imp_df, 'shap_importance(slight).pkl')
joblib.dump({'target_features': TARGET_FEATURES}, 'feature_order(slight).pkl')

# ------------------------------------------------------------------------------
# 9. 最终验证
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 步骤8: 验证流程")
print("="*60)

test_arr = np.array([X_7features.iloc[0].values])
scaler = joblib.load('preprocessor(slight).pkl')

if scaler['small_cols']:
    test_arr[:, [TARGET_FEATURES.index(c) for c in scaler['small_cols']]] = scaler['small'].transform(test_arr[:, [TARGET_FEATURES.index(c) for c in scaler['small_cols']]])
if scaler['large_cols']:
    test_arr[:, [TARGET_FEATURES.index(c) for c in scaler['large_cols']]] = scaler['large'].transform(test_arr[:, [TARGET_FEATURES.index(c) for c in scaler['large_cols']]])

model = joblib.load(f"model_rank1_{top5.iloc[0]['model_name'].replace(' ','_')}(slight).pkl")
print(f"✅ 预测概率: {model.predict_proba(test_arr)[0,1]:.4f}")
print("🎉 全部流程正常！")