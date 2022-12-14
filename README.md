# DengAI: Predicting Dengue Fever Outbreaks
By Oscar Schraenkler, Tizian Hamm & Anton Kleihues

- Description: Using environmental data collected by various U.S. Federal Government agencies—from the Centers for Disease Control and Prevention to the National Oceanic and Atmospheric Administration in the U.S. Department of Commerce— can we predict the number of dengue fever cases reported each week in San Juan, Puerto Rico and Iquitos, Peru?
- Data Source: https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/
- Type of project: Data Analysis, Regression modeling, Deep learning, Front-end production.

## Problem Statement
Dengue fever is a mosquito-borne disease that occurs in tropical parts of the world. In mild cases, symptoms are similar to the flu: fever, rash, and muscle and joint pain. In severe cases, dengue fever can cause severe bleeding, low blood pressure, and even death. Because it is carried by mosquitoes, the transmission dynamics of dengue are related to climate variables such as temperature and precipitation.

Our task is to predict the number of dengue cases each week in San Juan and Iquitos based on environmental variables describing changes in temperature, precipitation, vegetation, and more.

Accurate dengue fever predictions could help public health workers and local health authorities to better allocate resources towards fighting life threatening epidemics.

## Approach
We had two weeks to 
- [Exploratory Data Analysis](notebooks/Exploratory%20Data%20Analysis.ipynb) to understand and clean the data.
- [Regression models](notebooks/Regression%20Modeling.ipynb) and [deep learning models](notebooks/Deep%20Learning%20Models%20final%20notebook%20Tizian.ipynb) were optimized and evaluated.
- The best performing model, a random forest regressor, was submitted to the competition and scored 589 out of 12529 submissions, ranking among the top 4.8% of models submitted.
- This model was then put in production and a [front-end app](app.py) was created using Streamlit. 
- We made a [presentation](DengAI%20slides.pdf) to showcase our project.

## Our App
The app predicts dengue fever cases for the next two weeks in San Juan and Iquitos using a weather forecasting API. The app updates it's predictions daily with new environmental information.
Try out our app [HERE](https://oscarschraenkler-dengue-fever-predictions-app-ccawvn.streamlit.app/)!

## Future Work
- Gather data for other cities in order to expand app to predict cases in other cities.
- Collaborate with local health authorities to get access to recent data that can improve the accuracy of our model.
