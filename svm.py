import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # RandomForestClassifier 대신 SVC 임포트
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc, average_precision_score
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')