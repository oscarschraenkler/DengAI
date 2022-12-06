import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import requests

from sklearn.preprocessing import MinMaxScaler

#Models
from sklearn.ensemble import RandomForestRegressor


@st.cache
def peak_boost(preds, scalar, mixed=None):
    '''A function to boost peaks of the predictions'''
    #Take derivative of predictions
    deriv = np.gradient(preds)
    #Scale derivative
    new_deriv = deriv * scalar
    #Integrate the new derivative starting from 0
    integral = cumtrapz(new_deriv, initial=0)
    length = range(len(preds))
    #Add intercept to match initial range
    boosted = [x+preds[0] for x in integral]

    #Option ways to put floors on the output, preventing negative case counts
    if mixed == 'max':
        out = [max(boosted[i], preds[i]) for i in length]
    elif mixed == 'pos':
        out = [max(boosted[i], 0) for i in length]
    else:
        out = boosted

    #Doubling values above a 0.992 threshold in order to better estimate peak cases.
    Ser = pd.Series(out)

    quantile_value = Ser.quantile(q=0.99)
    print(f'Values are doubled over this threshold: {quantile_value}')

    for k,v in Ser.items():
        if (v > quantile_value):
            Ser[k] = int(Ser[k]*2)

    return np.array([int(x) for x in Ser])

@st.cache
def converter(x):
    return x-273.15

@st.cache
def preprocess_historical(data):

    #Reset index to prevent issues when merging
    reindexed_data = data.reset_index(drop = True)

    #Fill nas with interpolation
    features = reindexed_data.interpolate(method='linear')

    features = features[['precipitation_amt_mm', 'station_avg_temp_c',
           'reanalysis_dew_point_temp_k', 'station_max_temp_c','station_min_temp_c','reanalysis_relative_humidity_percent']]

    #Convert kelvin to celcius
    def converter(x):
        return x-273.15
    features['dew_point_temp_c'] = features['reanalysis_dew_point_temp_k'].map(converter)

    df = pd.DataFrame()
    df['precip'] = features['precipitation_amt_mm']
    df['temp'] = features['station_avg_temp_c']
    df['max_temp'] = features['station_max_temp_c']
    df['min_temp'] = features['station_min_temp_c']
    df['humidity'] = features['reanalysis_relative_humidity_percent']
    df['dew_point'] = features['dew_point_temp_c']

    #Add lagged features for 4 weeks
    to_shift = ['precip','temp',
       'max_temp','min_temp','humidity','dew_point']
    for i in to_shift:
        df[i+'_1lag'] = df[i].shift(+1)
        df[i+'_2lag'] = df[i].shift(+2)
        df[i+'_3lag'] = df[i].shift(+3)
        df[i+'_4lag'] = df[i].shift(+4)
    df = df.fillna(method='bfill')

    return df

@st.cache
def preprocess_api(url):

    data = requests.get(url).json()

    hourly_data = pd.DataFrame(data['hourly'])
    hourly_data['time'] = pd.to_datetime(hourly_data['time'])
    hourly_data = hourly_data.groupby(pd.Grouper(key="time", freq="1W")).mean()

    daily_data = pd.DataFrame(data['daily'])
    daily_data['time'] = pd.to_datetime(daily_data['time'])
    daily_data = daily_data.groupby(pd.Grouper(key="time", freq="1W")).mean()

    combined_data = hourly_data.join(daily_data)

    df = pd.DataFrame()
    df['precip'] = combined_data['precipitation_sum']
    df['temp'] = combined_data['temperature_2m']
    df['max_temp'] = combined_data['temperature_2m_max']
    df['min_temp'] = combined_data['temperature_2m_min']
    df['humidity'] = combined_data['relativehumidity_2m']
    df['dew_point'] = combined_data['dewpoint_2m']

    #Reset index
    final_df = df.reset_index(drop = True)

    #Add lagged features for 4 weeks
    to_shift = ['precip','temp',
       'max_temp','min_temp','humidity','dew_point']
    for i in to_shift:
        final_df[i+'_1lag'] = final_df[i].shift(+1)
        final_df[i+'_2lag'] = final_df[i].shift(+2)
        final_df[i+'_3lag'] = final_df[i].shift(+3)
        final_df[i+'_4lag'] = final_df[i].shift(+4)
    final_df = final_df.fillna(method='bfill')

    return final_df

data = pd.read_csv('raw_data/dengue_features_train.csv')
labels = pd.read_csv('raw_data/dengue_labels_train.csv')
data = data.merge(labels)
data_sj = data.iloc[:936,:]
data_iq = data.iloc[936:,:]
data_sj['dew_point_temp_c'] = data_sj['reanalysis_dew_point_temp_k'].map(converter)
sj_data = preprocess_historical(data_sj)
iq_data = preprocess_historical(data_iq)

sj_url = "https://api.open-meteo.com/v1/forecast?latitude=-3.75&longitude=-73.25&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m&models=best_match&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&start_date=2022-06-08&end_date=2022-12-18"
iq_url = "https://api.open-meteo.com/v1/forecast?latitude=-3.75&longitude=-73.25&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m&models=best_match&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&start_date=2022-06-08&end_date=2022-12-18"
sj_forecast = preprocess_api(sj_url)
iq_forecast = preprocess_api(iq_url)

params = {'max_depth': None,
 'max_features': 5,
 'min_samples_leaf': 6,
 'min_samples_split': 3,
 'n_estimators': 90,
 'warm_start': False}

iq_X = iq_data
iq_y = data_iq.total_cases
sj_X = sj_data
sj_y = data_sj.total_cases

@st.cache
def predict(params,iq_X,iq_y,sj_X,sj_y):
    #Initialize models
    sj_rf_model = RandomForestRegressor(**params, criterion='absolute_error')
    iq_rf_model = RandomForestRegressor(**params, criterion='absolute_error')

    #Fitting models on training data
    sj_rf_model.fit(sj_X, sj_y)
    iq_rf_model.fit(iq_X, iq_y)

    #Making predictions on training and testing data for San Juan
    sj_pred = sj_rf_model.predict(sj_forecast).astype(int)
    iq_pred = iq_rf_model.predict(iq_forecast).astype(int)
    return sj_pred,iq_pred

sj_pred = predict(params,iq_X,iq_y,sj_X,sj_y)[0]
iq_pred = predict(params,iq_X,iq_y,sj_X,sj_y)[1]

sj_boost = peak_boost(sj_pred, 4, 'max')
iq_boost = peak_boost(iq_pred, 2.5, 'max')

san_juan = sj_boost[-2:]
iquitos = iq_boost[-2:]


#st.form_submit_button('Make prediction')


City = st.sidebar.selectbox('Select City:', ('-','San Juan', 'Iquitos'))


if City=='San Juan':
    if (san_juan[0]+san_juan[1]) > 66:
        st.header('High Dengue Fever Risk Detected')
    elif (san_juan[0]+san_juan[1]) > 29 and iquitos[0] < 67:
        st.header('Medium Dengue Fever Risk Detected')
    else:
        st.header('Low Dengue Fever Risk Detected')

    col1, col2= st.columns(2)

    with col1:
        st.header("Next week:")
        st.header(f'{san_juan[0]} cases')
    with col2:
        st.header("In two weeks:")
        st.header(f'{san_juan[1]} cases')
elif City=='Iquitos':
    st.header('Iquitos')
    st.line_chart(data=iquitos)
    st.text(f'IQ Week 1: {iquitos[0]}')
    st.text(f'IQ Week 2: {iquitos[1]}')
    if (iquitos[0]+iquitos[1]) > 16:
        st.text('High Risk')
    elif (iquitos[0]+iquitos[1]) > 5 and (iquitos[0]+iquitos[1]) < 17:
        st.text('Medium Risk')
    else:
        st.text('Low Risk')
else:
    header= st.container()
    with header:
        st.title('Dengue Fever Forecast')
    explanation=st.container()
    dataset= st.container()
    facts= st.container()
    features=st.container()
    model_training=st.container()
    with explanation:
        st.header('Explanation of our App/ model')
        st.text('Explanation of why our predicitions are important...')
    with facts:
        st.header('Facts about Dengue Fever!')
        st.text('basic background facts about dengue...')
    with features:
        st.header('Most important features:')
        st.text('Explanation of features..')
