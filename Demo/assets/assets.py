from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

CAT_VAL = ['0_MaGD', '11_MaGD', '15_MaGD', '1_MaGD', 
        '2_MaGD', '3_MaGD', '4_MaGD', '6_MaGD', 
        '7_MaGD', '8_MaGD', '00_PGD', '02_PGD', 
        '03_PGD', '05_PGD', '06_PGD', '07_PGD', 
        '08_PGD', 'DA_NghiepVuChiTiet', 
        'SA_NghiepVuChiTiet', 'GIOITINH', 'Account_Management']

CON_VAL = ['avg_VND', 'min_VND', 'max_VND', 'avg_interest']

algorithms_dict = {
    'Logistics Regression': LogisticRegression(), 
    'Decision Tree': DecisionTreeClassifier(), 
    'Random Forest': RandomForestClassifier(n_estimators=500, n_jobs=-1), 
    'XGBoost': XGBClassifier(n_jobs=-1)
    }