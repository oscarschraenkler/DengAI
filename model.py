import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
def converter(x):
    return x-273.15

def preprocess_historical(data):

    #Reset index to prevent issues when merging
    reindexed_data = data.reset_index(drop = True)

    #Fill nas with interpolation
    features = reindexed_data.interpolate(method='linear')

    features = features[['precipitation_amt_mm', 'station_avg_temp_c',
           'reanalysis_dew_point_temp_k', 'station_max_temp_c','station_min_temp_c','reanalysis_relative_humidity_percent']]

    #Convert kelvin to celcius
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

data = pd.read_csv('raw_data/dengue_features_train.csv')
labels = pd.read_csv('raw_data/dengue_labels_train.csv')
data = data.merge(labels)
data_sj = data.iloc[:936,:]
data_iq = data.iloc[936:,:]
sj_data = preprocess_historical(data_sj)
iq_data = preprocess_historical(data_iq)

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

#Initialize models
sj_rf_model = RandomForestRegressor(**params, criterion='absolute_error')
iq_rf_model = RandomForestRegressor(**params, criterion='absolute_error')

#Fitting models on training data
sj_rf_model.fit(sj_X, sj_y)
iq_rf_model.fit(iq_X, iq_y)

pickle_out = open("sj_model.pkl", "wb")
pickle.dump(sj_rf_model, pickle_out)
pickle_out.close()

pickle_out = open("iq_model.pkl", "wb")
pickle.dump(iq_rf_model, pickle_out)
pickle_out.close()
