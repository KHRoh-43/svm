import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # RandomForestClassifier 대신 SVC 임포트
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc, average_precision_score
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

data_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
col_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
try:
    data=pd.read_csv(data_url, header=None, names=col_names)
    print("데이터 로드 완료. (총 {}개 샘플)",format(len(data)))
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    print("로컬 환경에서 직접 CSV 파일을 다운로드하여 경로를 지정하거나, 데이터셋을 확인해 주세요.")
    exit()

cols_to_check_for_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data_cleaned = data.copy()
data_cleaned[cols_to_check_for_zeros] = data_cleaned[cols_to_check_for_zeros].replace(0, np.nan)

data_cleaned.dropna(inplace=True)
X=data_cleaned.drop('Outcome', axis=1)
Y=data_cleaned['Outcome']
print(f"결측치(0) 포함 행 제거 후, 최종 {len(X)}개 샘플 사용.")

def run_svm_experiment(C, kernel, gamma, test_size):
    """
    Args:
        C (float): parameter for soft margin
        kernel (str): kernel type ('linear', 'rbf', 'poly').
        gamma (str or float): RBF 커널 게수.
        test_size (float): 테스트 데이터셋의 비율 (0.0 ~ 0.1).
    """
    X_train, X_test, y_train, y_test = train_test_split(S, y, test_size=test_size,random_state=42,stratify=y)
    train_ratio=1-test_size
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("-" * 60)
    print(f"** 실험 조건 **")
    print(f"데이터 분할 비율 (Train/Test): {teain_ratio*100:.0f}% / {test_size*100:.0f}%")
    print(f"soft margin penalty (C): {C}")
    print(f"커널 (kernel): {kernel}")
    print(f"감마 (gamma): {gamma}")
    print("-" * 60)

    try:
        svm_model=SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=2025
        )
        svm_model.fit(X_train_scaled, y_train)
        y_pred = svm_model.predict(X_test_scaled)
        y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]  
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        print("Confusion Matrix(혼동 행렬):")
        print(" [TN: (실제 N, 에측 N), FP: (실제 N, 예측 P)]")
        print(" [FN: (실제 P, 예측 N), TP: (실제 P, 예측 P)]")
        print(cm)
        print(f"\nF1 Score: {f1:.4f}")
        print(f"Accuracy: {acc:.4f}\n")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc9fpr, tpr)

        plt.figure(figsize=(12, 5))

        plt.plot(fpr, tpr, color='navy', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.show()

    except Exception as e:
        print(f"SVM 모델 학습/평가 중 오류 발생: {e}")
        print("입력값을 확인해 주세요. C는 0보다 커야 하며, kernel/gamma 값을 확인해 주세요.")

    #====================================================
    #====================================================

    run_svm_experiment(
        C=0.1,
        kernel='rbf',
        gamma='scale',
        test_size=0.1
    )

