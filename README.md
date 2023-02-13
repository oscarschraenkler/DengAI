# DengAI: Predicting Dengue Fever Outbreaks
By Oscar Schraenkler, Tizian Hamm & Anton Kleihues

- Description: Using environmental data collected by various U.S. Federal Government agencies—from the Centers for Disease Control and Prevention to the National Oceanic and Atmospheric Administration in the U.S. Department of Commerce— can we predict the number of dengue fever cases reported each week in San Juan, Puerto Rico and Iquitos, Peru?
- Data Source: https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/
- Type of project: Data Analysis, Regression modeling, Deep learning, Front-end production.

## Problem Statement
Dengue fever is a disease transmitted by mosquitoes that primarily affects tropical regions of the world. Its symptoms can range from mild flu-like symptoms, such as fever, rash, and muscle and joint pain, to more severe complications such as bleeding, low blood pressure, and even death. As mosquitoes thrive in hot and humid conditions, the transmission of dengue fever is closely tied to climate variables such as temperature and precipitation.

Our objective is to predict the number of dengue cases that will occur in San Juan and Iquitos each week based on environmental variables, such as changes in temperature, precipitation, and vegetation.

By providing accurate predictions of dengue fever cases, we could help public health workers and local health authorities to better allocate resources towards reducing the impact of potentially life-threatening epidemics.

## Approach
We had a timeframe of two weeks to explore the data, develop a model and put it in production. Here were our steps:
- An [exploratory data analysis](notebooks/Exploratory%20Data%20Analysis.ipynb) was conducted to understand and clean the data.
- [Regression models](notebooks/Regression%20Modeling.ipynb) and [deep learning models](notebooks/Deep%20Learning%20Models%20Tizian.ipynb) were optimized and evaluated to find the best model.
- The best performing model, a random forest regressor, was submitted to the competition and ranked 589 out of 12529 submissions, scoring among the top 4.8% of models submitted.
- This model was then put in production and a [front-end app](app.py) was created using Streamlit. 
- We [presented](https://www.youtube.com/watch?v=NWceVsPYP7g) our project. View the slides [here](DengAI%20slides.pdf).

## Our App
The app predicts dengue fever cases for the next two weeks in San Juan and Iquitos using a weather forecasting API. The app updates it's predictions daily with new environmental information.
Try out our app [HERE](https://oscarschraenkler-dengue-fever-predictions-app-ccawvn.streamlit.app/)!

## Future Work
- Gather data for other cities in order to expand app to predict cases in other cities.
- Collaborate with local health authorities to get access to recent data that can improve the accuracy of our model.

Read about the details of our project on [Medium](https://medium.com/@o.schraenkler/dengai-predicting-dengue-fever-outbreaks-56b201e55983). Watch the presentation on [YouTube](https://www.youtube.com/watch?v=NWceVsPYP7g).
