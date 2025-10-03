import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Library for data mining
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import train_test_split
# Library models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
from xgboost import XGBClassifier
# Library for metrics


import time
import gc
from PIL import Image
from typing import Tuple, Any

import warnings
warnings.filterwarnings("ignore")


from Demo.assets.assets import CAT_VAL, CON_VAL, algorithms_dict
def make_log_column(df: pd.DataFrame, vals: list, prefix: bool = False) -> Tuple[pd.DataFrame, Any]:
    scaler = 'none'
    for val in vals:
        if prefix == True:
            val_name = val + '_log'
        else:
            val_name = val
        df.loc[:, val_name] = np.log(df[val])
        # Replace values that smaller than 0 to 0.
        df.loc[:, val_name][df.loc[:, val_name] < 0] = 0
    return df, scaler


def make_RobustScaler_column(df: pd.DataFrame, vals: list, prefix: bool = False)-> Tuple[pd.DataFrame, Any]:
    scaler = RobustScaler()
    robust_scaler = scaler.fit_transform(df[vals])
    if prefix == True:
        val_name = [x + '_Rscale' for x in vals]
    else:
        val_name = vals
    tmp = pd.DataFrame(robust_scaler, columns=val_name)
    for val in val_name:
        df.loc[:, val] = tmp[val].values
    return df, scaler

def make_StandardScaler_column(df: pd.DataFrame, vals: list, prefix=False):
    scaler = StandardScaler()
    standard_scaler = scaler.fit_transform(df[vals])
    if prefix == True:
        val_name = [x + '_Stscale' for x in vals]
    else:
        val_name = vals   
    tmp = pd.DataFrame(standard_scaler, columns=val_name)
    for val in val_name:
        df.loc[:, val] = tmp[val].values
    return df, scaler



def model(df: pd.DataFrame, algorithm: str, scaling: str ='None', imb_handling='None'):

    scaler = 'none'
    if scaling == 'Logarithm':
        df, scaler = make_log_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log' in val and 'USD' not in val and '_Stscale' not in val and '_Rscale' not in val]
    elif scaling == 'Standard Scaler':
        df, scaler = make_StandardScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_Stscale' in val and 'USD' not in val]
    elif scaling == 'Robust Scaler':
        df, scaler = make_RobustScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_Rscale' in val and 'USD' not in val]
    elif scaling == 'log_Stscale':
        df = make_log_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log' in val and 'USD' not in val and '_Stscale' not in val and '_Rscale' not in val]
        df, scaler = make_StandardScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log_Stscale' in val and 'USD' not in val]
    elif scaling == 'log_Rscale':
        df = make_log_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log' in val and 'USD' not in val and '_Stscale' not in val and '_Rscale' not in val]
        df, scaler = make_StandardScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log_Rscale' in val and 'USD' not in val]
    
    inputs = CON_VAL + CAT_VAL
    
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if imb_handling.lower() == 'utils':
        df_train = X_train.merge(y_train, on=X_train.index)
        df_0 = df_train.loc[df_train['label'] == 0]
        df_1 = df_train.loc[df_train['label'] == 1]     
   
        df_1_resample = resample(df_1, replace=True, n_samples = df_0.shape[0])
        
        df_upsampled = pd.concat([df_1_resample, df_0])
        X_train = df_upsampled.drop('label', axis=1)
        y_train = df_upsampled['label']
        
    elif imb_handling.lower() == 'smote':
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)       
    
    X_train, X_test = X_train[inputs], X_test[inputs]
    


    model_ = algorithms_dict[algorithm].fit(X_train, y_train)
    gc.enable()
    del df, y_train, X_test, y_test
    gc.collect()
    return model_, scaler, X_train.columns

def evaluate_model(df: pd.DataFrame, algorithm: str, scaling: str = 'None', imb_handling: str ='None'):
    # Conditions for scaler
    if scaling == 'Logarithm':
        df, scaler = make_log_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log' in val and 'USD' not in val and '_Stscale' not in val and '_Rscale' not in val]
    elif scaling == 'Standard Scaler':
        df, scaler = make_StandardScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_Stscale' in val and 'USD' not in val]
    elif scaling == 'Robust Scaler':
        df, scaler = make_RobustScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_Rscale' in val and 'USD' not in val]
    elif scaling == 'log_Stscale':
        df = make_log_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log' in val and 'USD' not in val and '_Stscale' not in val and '_Rscale' not in val]
        df, scaler = make_StandardScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log_Stscale' in val and 'USD' not in val]
    elif scaling == 'log_Rscale':
        df = make_log_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log' in val and 'USD' not in val and '_Stscale' not in val and '_Rscale' not in val]
        df, scaler = make_StandardScaler_column(df, CON_VAL, prefix=True)
        CON_VAL = [val for val in df.columns.tolist() if '_log_Rscale' in val and 'USD' not in val]
    
    inputs = CON_VAL + CAT_VAL

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Conditions for Resampling
    if imb_handling.lower() == 'utils':
        df_train = X_train.merge(y_train, on=X_train.index)
        df_0 = df_train.loc[df_train['label'] == 0]
        df_1 = df_train.loc[df_train['label'] == 1]     
   
        df_1_resample = resample(df_1, replace=True, n_samples = df_0.shape[0])
        
        df_upsampled = pd.concat([df_1_resample, df_0])
        X_train = df_upsampled.drop('label', axis=1)
        y_train = df_upsampled['label']
    elif imb_handling.lower() == 'smote':
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)       
    
    X_train, X_test = X_train[inputs], X_test[inputs]
    
    # Fitting model
    model_ = algorithms_dict[algorithm].fit(X_train, y_train)
    score_test = model_.score(X_test, y_test)
    score_train = model_.score(X_train, y_train)
    # Evalutate factors    
    predictions = model_.predict(X_test)
    recall = recall_score(y_test, predictions) * 100
    f1 = f1_score(y_test, predictions) * 100
    cfs_matrix = confusion_matrix(y_test, predictions)
    class_rep = classification_report(y_test, predictions, output_dict=True)
   
    gc.enable()
    # del X_train, y_train, X_test, y_test
    gc.collect()

    return f1, recall, score_test, score_train, CON_VAL, CAT_VAL, cfs_matrix, class_rep

'------------------------------------------------------------------------------------------------'

st.header('= = = = = = = = = = = = = = = = = = = = = = = = = = = =')
st.markdown("<h1 style='text-align: center;'>CUSTOMER CHURN DEMO</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>PREDICTING WHETHER A CUSTOMER IS GOING TO WITHDRAW OR NOT (BETTER NOT!)</h2>", unsafe_allow_html=True)
st.header('= = = = = = = = = = = = = = = = = = = = = = = = = = = =')
image1 = Image.open('Demo/assets/photo-1534951009808-766178b47a4f.jpg')
st.image(image1, use_column_width=True)

'-----------------------------------------------------------------------------------------------'
page = st.sidebar.selectbox('Please choose a page', ['Explore data', 'Predict your data'])

if page == 'Predict your data':
    st.title('This page is for prediction your data.')

    st.header('Which of the Label method do you want to run on?')
    data_bt = st.radio('Label:', ['Experienced based Labelling', 'Propagating Labelling'])
    df = pd.read_csv('Demo/assets/Datas/data_cleaned.csv', index_col='Unnamed: 0')
    df2 = pd.read_csv('Demo/assets/Datas/data_cleaned_propag_label.csv', index_col='Unnamed: 0')
    if data_bt == 'Experienced based Labelling':
        df = df
    else:
        df = df2

    st.write('Please select your desire scaler type.')
    scaler_bt = st.radio('Scaler type:', ['None', 'Logarithm', 'Standard Scaler', 'Robust Scaler'])

    st.write('Please select your desire sampling method.')
    imh_bt = st.radio('Sampling method', ['None', 'Smote', 'Utils'])

    st.write('Please select your desire algorithm.')
    algo_bt = st.radio('Algorithms', ['Logistics Regression', 'Decision Tree', 'Random Forest', 'XGBoost'])

    st.markdown(f'''This model is run with: <br>\n
                    * scaler type: {scaler_bt},
                    * sampling method: {imh_bt},
                    * algorithm: {algo_bt}.''', unsafe_allow_html=True)

    model_, scaler, cols = model(df=df, scaling=scaler_bt, imb_handling=imh_bt, algorithm=algo_bt)

    X_predict = pd.read_csv('Demo/assets/Datas/data_test1.csv')
    X_predict = X_predict.apply(pd.to_numeric)

    if scaler == 'none':
        X_predict = X_predict, scaler = make_log_column(X_predict, 
                                                        [x for x in X_predict.columns\
                                                        if len(X_predict[x].unique()) >= 3],
                                                        prefix=False)
    else:
        X_predict.iloc[:,:4] = scaler.transform(np.array(X_predict.iloc[:,:4]))

    X_predict.columns = cols
    image2 = Image.open('Demo/assets/photo-1444653614773-995cb1ef9efa.jpg')
    st.image(image2, use_column_width=True)
    btn = st.button("Predict!")
    if btn:
        my_table = st.table(X_predict.head(10))
        df_predict = pd.DataFrame(model_.predict_proba(X_predict))
        df_predict.loc[:, 'label'] = model_.predict(X_predict)
        st.write('Prediction')
        st.write(df_predict)
    
    btn3 = st.button("Game!")
    if btn3:
        X_game = pd.read_csv('Demo/assets/Datas/data_test2.csv')
        X_game = X_game.apply(pd.to_numeric)
        X_game.columns = cols
        try:    
            if scaler == 'none':
                X_game = X_game, scaler = make_log_column(X_game, 
                                                        [x for x in X_game.columns\
                                                        if len(X_game[x].unique()) >= 3],
                                                        prefix=False)
            else:
                X_game.iloc[:,:4] = scaler.transform(np.array(X_game.iloc[:,:4]))
            st.dataframe(X_game)
            st.write('Prediction')
            df_predict_game = pd.DataFrame(model_.predict_proba(X_game))
            df_predict_game.loc[:, 'label'] = model_.predict(X_game)
            st.write(df_predict_game)
        except Exception as error:
            st.write('None row entered. Please enter your data.')
else:
    st.title('This page is for exploring and evaluating your model.')

    st.header('Which of the Label method do you want to run on?')
    data_bt = st.radio('Label:', ['Experienced based Labelling', 'Propagating Labelling'])
    
    df = pd.read_csv('Demo/assets/Datas/data_cleaned.csv', index_col='Unnamed: 0')
    df2 = pd.read_csv('Demo/assets/Datas/data_cleaned_propag_label.csv', index_col='Unnamed: 0')
    if data_bt == 'Experienced based Labelling':
        df = df
    else:
        df = df2
    btn1 = st.button("Show me some data!")
    if btn1:    
        st.dataframe(df.head(10))
    
    st.write('Please select your desire scaler type.')
    scaler_bt = st.radio('Scaler type:', ['None', 'Logarithm', 'Standard Scaler', 'Robust Scaler'])

    st.write('Please select your desire sampling method.')
    imh_bt = st.radio('Sampling method:', ['None', 'Smote', 'Utils'])

    st.write('Please select your desire algorithm.')
    algo_bt = st.radio('Algorithms:', ['Logistics Regression', 'Decision Tree', 'Random Forest', 'XGBoost'])

    st.markdown(f'''This model is run with <br>\n
                    * scaler type: {scaler_bt},
                    * sampling method: {imh_bt},
                    * algorithm: {algo_bt}.''', unsafe_allow_html=True)

    
    t1 = time.perf_counter()
    f1, recall, score_test, score_train, con, cat, cfs_matrix, class_rep = evaluate_model(df=df, scaling=scaler_bt, imb_handling=imh_bt, algorithm=algo_bt)
    t2 = time.perf_counter()
    image3 = Image.open('Demo/assets/photo-1527474305487-b87b222841cc.jpg')
    st.image(image3, use_column_width=True)
  

    st.header('Some basic Information')
    st.write('Continuous: ', str(con))
    st.write('Categorical: ', str(cat))

    st.header("Let's plot some..")
    var = st.text_input('Enter your variable here:') 

    if var:
        if var in con:
            interval_dict = {
                'max_VND': 900000, 'min_VND': 9000, 'avg_VND': 90000, 'avg_interest': 0.0001,
                'max_VND_log': 0.01, 'min_VND_log': 0.01, 'avg_VND_log': 0.01, 'avg_interest_log': 0.095,
                'max_VND_Stscale': 1, 'min_VND_Stscale': 1.01, 'avg_VND_Stscale': 1.11, 'avg_interest_Stscale': 0.01,
                'max_VND_Rscale': 50, 'min_VND_Rscale': 50, 'avg_VND_Rscale': 50, 'avg_interest_Rscale': 0.01
                }
            fig = plt.figure()
            sns.distplot(df[var], kde_kws={'bw': interval_dict[var]}, color='r')
            plt.title(f'Distribution plot {var}')
            st.pyplot(fig, scale=True)
        elif var in cat:
            fig = plt.figure()
            df[var].value_counts().plot.bar(color='#ffa600', alpha=0.8)
            plt.title(f'{var} count plot')
            plt.grid(axis='y')
            st.pyplot(fig, scale=True)    
        else:
            pass        
    
    btn2 = st.button("Run model!")
    if btn2: 
        st.write('f1-score: ', f1)
        st.write('recall-score: ', recall)
        st.write('Accuracy score test: ', score_test)
        st.write('Accuracy score train: ', score_train)
        st.write('Time running:', t2 - t1, 'seconds.')
        st.markdown('Confusion matrix')
        st.write(cfs_matrix)
        st.markdown('Classification report')
        st.dataframe(class_rep)