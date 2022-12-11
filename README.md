# DengAI: Predicting Dengue Fever Outbreaks

- Description: Using environmental data collected by various U.S. Federal Government agencies—from the Centers for Disease Control and Prevention to the National Oceanic and Atmospheric Administration in the U.S. Department of Commerce— can we predict the number of dengue fever cases reported each week in San Juan, Puerto Rico and Iquitos, Peru?
- Data Source: https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/
- Type of analysis: Data Analysis, Regression modeling and Deep learning

## Problem Statement
Dengue fever is a mosquito-borne disease that occurs in tropical parts of the world. In mild cases, symptoms are similar to the flu: fever, rash, and muscle and joint pain. In severe cases, dengue fever can cause severe bleeding, low blood pressure, and even death. Because it is carried by mosquitoes, the transmission dynamics of dengue are related to climate variables such as temperature and precipitation.

Our task is to predict the number of dengue cases each week in San Juan and Iquitos based on environmental variables describing changes in temperature, precipitation, vegetation, and more.

Accurate dengue fever predictions could help public health workers and local health authorities to better allocate resources towards fighting life threatening epidemics.

## Approach
- [Exploratory Data Analysis](notebooks/Exploratory%20Data%20Analysis.ipynb).
- [Regression models](notebooks/Regression%20Modeling.ipynb) and [deep learning models](notebooks/Deep%20Learning%20Models%20final%20notebook%20Tizian.ipynb) optimized and evaluated.
- Best model put in production and [front-end app](app.py) created using Streamlit.

## Our App
Our app predicts the dengue fever cases for the next two weeks in San Juan and Iquitos using a weather forecasting API. The app updates it's predictions daily with new weather information.
See our app here: https://oscarschraenkler-dengue-fever-predictions-app-ccawvn.streamlit.app/
