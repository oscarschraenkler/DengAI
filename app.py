import streamlit as st

import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
import pickle
from PIL import Image
import requests
from datetime import date
from datetime import timedelta
from streamlit_lottie import st_lottie
import json

st.set_page_config(page_title="Dengue Fever Forecast",
                   page_icon="😷")

header1, header2= st.columns(2)
facts= st.container()
explanation=st.container()
predict1 = st.container()
with header1:
    st.title('Dengue Fever Forecast')
    st.markdown('Created by Oscar Schraenkler, Anton Kleihues & Tizian Hamm')
with header2:
    image = Image.open('image/dengai_new.PNG')
    st.image(image, width=180)
with explanation:
    st.header('The Model')
    st.markdown('The model predicts the number of dengue cases for the next two weeks in San Juan (Puerto Rico) and Iquitos (Peru) using a weather forecasting API describing changes in temperature, precipitation, humidity, and more.')
    with st.expander('Read more...'):
        st.write("A random forest regressor was trained on over 20 years of weekly climate-related data collected in San Juan and Iquitos. A peak boosting method implementing calculus functions was then mapped on to the predictions to exaggerate the large outbreaks, increasing the accuracy of our predictions.")
        st.write("**Most Important Features:**")
        st.markdown('* **Humidity:** The concentration of water vapor present in the air.')
        st.markdown('* **Dew point temperature:** The temperature to which air must be cooled to become saturated with water vapor, assuming constant air pressure and water content.')
        st.markdown('* **Air temperature:** The temperature of the air in a location.')
        st.markdown('* **Precipitation:** Any product of the condensation of atmospheric water vapor that falls under gravitational pull from clouds.')

with facts:
    st.header('About Dengue Fever')
    st.markdown('Dengue Fever is a mosquito-borne virus that occurs in tropical regions of the world that can cause high fever, headaches, vomiting, a characteristic skin itching and skin rash, and even death.')
    with st.expander('Read more...'):
        st.markdown('The global incidence of dengue has grown dramatically with about half of the worlds population now at risk, and is estimated to cause a global economic burden of $8.9 billion per year.')
        st.markdown('The transmission dynamics of dengue are related to climate variables such as temperature and humidity. An understanding of the relationship between climate and dengue dynamics could improve research initiatives and resource allocation to help fight life-threatening pandemics.')
        st.markdown('**Dengue cases distribution throughout the year in San Juan, Puerto Rico and Iquitos, Peru:**')
        image = Image.open('image/distribution.png')
        st.image(image)
def load_lottieurl(url: str):
    r= requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
with predict1:
    st.header('Choose a city to get case predictions:')

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

from datetime import date
# Returns the date parameters
today = date.today()
future = today + timedelta(days=14)
start = today  - timedelta(days=180)
future = str(future)
start = str(start)
sj_url = f"https://api.open-meteo.com/v1/forecast?latitude=-3.75&longitude=-73.25&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m&models=best_match&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&start_date={start}&end_date={future}"
iq_url = f"https://api.open-meteo.com/v1/forecast?latitude=-3.75&longitude=-73.25&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m&models=best_match&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&start_date={start}&end_date={future}"
sj_forecast = preprocess_api(sj_url)
iq_forecast = preprocess_api(iq_url)

pickle_in = open('iq_model.pkl', 'rb')
iq_rf_model = pickle.load(pickle_in)

pickle_in = open('sj_model.pkl', 'rb')
sj_rf_model = pickle.load(pickle_in)

@st.cache
def predict(model,input):
    pred = model.predict(input).astype(int)
    return pred

sj_pred = predict(sj_rf_model,sj_forecast)
iq_pred = predict(iq_rf_model,iq_forecast)

sj_boost = peak_boost(sj_pred, 4, 'max')
iq_boost = peak_boost(iq_pred, 2.5, 'max')

san_juan = sj_boost[-2:]
iquitos = iq_boost[-2:]

stop1 = today  + timedelta(days=7)
stop2 = stop1 + timedelta(days=7)
stop1 = str(stop1)
stop2 = str(stop2)
today = str(today)

City = st.selectbox('Select City:', ('-','San Juan, Puerto Rico', 'Iquitos, Peru'))

if City=='San Juan, Puerto Rico':
    if (san_juan[0]+san_juan[1]) > 66:
        st.header('High Dengue Fever Risk in San Juan 🚨')
    elif (san_juan[0]+san_juan[1]) > 29 and iquitos[0] < 67:
        st.header('Medium Dengue Fever Risk in San Juan ⚠️')
    else:
        st.header('Low Dengue Fever Risk in San Juan ✅')

    col1, col2= st.columns(2)

    with col1:
        st.text(f"Week starting {today}:")
        st.header(f'{san_juan[0]} cases')
    with col2:
        st.text(f"Week starting {stop1}:")
        st.header(f'{san_juan[1]} cases')
elif City=='Iquitos, Peru':
    if (iquitos[0]+iquitos[1]) > 16:
        st.header('High Dengue Fever Risk in Iquitos 🚨')
    elif (iquitos[0]+iquitos[1]) > 5 and (iquitos[0]+iquitos[1]) < 17:
        st.header('Medium Dengue Fever Risk in Iquitos ⚠️')
    else:
        st.header('Low Dengue Fever Risk in Iquitos ✅')

    col1, col2= st.columns(2)

    with col1:
        st.text(f"Week starting {today}:")
        st.header(f'{iquitos[0]} cases')
    with col2:
        st.text(f"Week starting {stop1}:")
        st.header(f'{iquitos[1]} cases')
else:
    block1, block2, block3= st.columns(3)
    with block2:
        lottie_mosquito= load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_trge7v5t.json')
        st_lottie(
        lottie_mosquito,
        speed=1,
        reverse=False,
        width=100,
        quality="low")
