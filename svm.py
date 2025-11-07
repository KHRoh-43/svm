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