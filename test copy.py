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
df = pd.read_excel('ReadText_parkinson_features.xlsx')
target = 'status'
exclude_cols = ['idx', 'status', 'file']

X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
y = df[target]

# 性别编码
if 'gender' in X.columns:
    X['gender'] = X['gender'].apply(lambda x: 1 if str(x).lower() in ['男', 1] else 0)

# 缺失值填充
X = X.fillna(X.mean())

# ------------------------------------------------------------------------------
# 2. 分范围标准化（修改版）
# ------------------------------------------------------------------------------
small_scale_cols = [c for c in X.columns if X[c].max() <= 10]
large_scale_cols = [c for c in X.columns if X[c].max() > 10]

# 先fit，但不transform X
small_scaler = StandardScaler()
large_scaler = RobustScaler()

if small_scale_cols:
    small_scaler.fit(X[small_scale_cols])
if large_scale_cols:
    large_scaler.fit(X[large_scale_cols])

# 保存预处理器（包含fit信息）
preprocessor = {
    'small': small_scaler,
    'large': large_scaler,
    'small_cols': small_scale_cols,
    'large_cols': large_scale_cols,
    # 👇 新增：保存所有特征名（关键！）
    'all_features': X.columns.tolist()
}

# 再对X进行标准化（用于后续训练）
if small_scale_cols:
    X[small_scale_cols] = small_scaler.transform(X[small_scale_cols])
if large_scale_cols:
    X[large_scale_cols] = large_scaler.transform(X[large_scale_cols])

joblib.dump(preprocessor, 'preprocessor.pkl')
print("✅ 预处理已保存：preprocessor.pkl（包含特征名信息）")


# ------------------------------------------------------------------------------
# 3. 分类专用 LASSO 特征筛选
# ------------------------------------------------------------------------------
print("⏳ 正在运行【分类专用LASSO】特征筛选...")

lasso_clf = LogisticRegressionCV(
    cv=5,
    penalty='l1',
    solver='liblinear',
    class_weight='balanced',
    max_iter=10000,
    random_state=42
)
lasso_clf.fit(X, y)

selector = SelectFromModel(lasso_clf, threshold=1e-5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
X = pd.DataFrame(X_selected, columns=selected_features)

print("\n" + "="*60)
print("📌 LASSO 筛选后保留的特征：")
print("="*60)
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2d}. {feat}")
print(f"\n✅ 最终保留特征数：{len(selected_features)}")

# ------------------------------------------------------------------------------
# 4. 留一交叉验证 LOOCV
# ------------------------------------------------------------------------------
loocv = LeaveOneOut()
X_ = X.values
y_ = y.values

# ------------------------------------------------------------------------------
# 5. 模型列表
# ------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    # "Ridge Classifier": RidgeClassifier(random_state=42),
    # "KNN": KNeighborsClassifier(),
    # "SVC RBF": SVC(kernel='rbf', probability=True, random_state=42),
    # "Linear SVC": LinearSVC(random_state=42),
    # "Decision Tree": DecisionTreeClassifier(random_state=42),
    # "Random Forest": RandomForestClassifier(random_state=42),
    # "Extra Trees": ExtraTreesClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    # "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    # "Bagging": BaggingClassifier(random_state=42),
    # "Gaussian NB": GaussianNB(),
    "Bernoulli NB": BernoulliNB(),
    # "LDA": LinearDiscriminantAnalysis(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    # "MLP": MLPClassifier(random_state=42, max_iter=500),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    #"LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    #"CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    # "SGD Classifier": SGDClassifier(random_state=42),
    # "SVC Linear": SVC(kernel='linear', probability=True, random_state=42),
    # "SVC Poly": SVC(kernel='poly', probability=True, random_state=42),
    "Nearest Centroid": NearestCentroid()
}

# ------------------------------------------------------------------------------
# 6. LOOCV 训练所有模型
# ------------------------------------------------------------------------------
results = []
print("\n" + "="*60)
print("🔥 留一交叉验证 LOOCV 训练中...")
print("="*60)

for name, model in models.items():
    try:
        y_pred = cross_val_predict(model, X_, y_, cv=loocv)
        y_prob = cross_val_predict(model, X_, y_, cv=loocv, method='predict_proba')[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_, y_pred)
        f1 = f1_score(y_, y_pred)
        auc = roc_auc_score(y_, y_prob) if y_prob is not None else acc

        model.fit(X_, y_)
        results.append({
            "model_name": name,
            "model": model,
            "acc": acc,
            "f1": f1,
            "auc": auc
        })
        print(f"{name:<20} | AUC={auc:.3f} | Acc={acc:.3f}")
    except Exception as e:
        continue

# ------------------------------------------------------------------------------
# 7. 输出 TOP5 模型
# ------------------------------------------------------------------------------
df_res = pd.DataFrame(results).sort_values('auc', ascending=False).reset_index(drop=True)
top5 = df_res.head(5)

print("\n" + "="*60)
print("🏆 留一交叉验证最优 5 个模型")
print("="*60)
for i, row in top5.iterrows():
    print(f"{i+1}. {row['model_name']:18} | AUC={row['auc']:.3f} | Acc={row['acc']:.3f}")

# ------------------------------------------------------------------------------
# 8. ✅ 保存 TOP5 模型为【单独 .pkl】（后端直接用）
# ------------------------------------------------------------------------------
for idx, row in top5.iterrows():
    model_name = row["model_name"].replace(" ", "_")
    joblib.dump(row["model"], f"model_rank{idx+1}_{model_name}.pkl")

print("\n✅ TOP5 模型已全部保存为独立 .pkl 文件！")

# ------------------------------------------------------------------------------
# 9. 特征重要性
# ------------------------------------------------------------------------------
best_model = top5.iloc[0]['model']

if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    try:
        feat_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(best_model.coef_[0])
        }).sort_values('importance', ascending=False)
    except:
        feat_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': 1.0
        })

joblib.dump(feat_imp, 'feature_importance.pkl')

# ------------------------------------------------------------------------------
# 10. SHAP 重要性（只输出前 6）
# ------------------------------------------------------------------------------
import shap
print("\n" + "="*60)
print("📊 SHAP 特征重要性 TOP7（直接复制到论文）")
print("="*60)

explainer = shap.Explainer(best_model.predict, X)
shap_values = explainer(X)
shap_importance = np.abs(shap_values.values).mean(axis=0)

shap_imp_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

# ✅ 只打印前 7 个
for i, (feat, imp) in enumerate(zip(shap_imp_df['feature'], shap_imp_df['shap_importance']), 1):
    if i > 7: break
    print(f"{i}. {feat}")

joblib.dump(shap_imp_df, 'shap_importance.pkl')
print("\n✅ shap_importance.pkl 已保存")