#######################
##                   ##
##  FINAL PROJECT  ##
##  CMSE 830         ##
##  FOREST FIRES APP ##
##                   ##
#######################

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew
import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import numpy as np
import csv
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (r2_score,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             mean_absolute_error,
                             mean_squared_log_error)
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import (AdaBoostRegressor,
                              RandomForestRegressor,
                              ExtraTreesRegressor, 
                              GradientBoostingRegressor)
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  Lasso,)

# read data
mpg_df = pd.read_csv("auto-mpg.csv", na_values = "?")


add_selectbox = st.sidebar.selectbox(
    "Please Select A Chapter",
    ("Introduction of data", "Check Dataset and EDA", "ML step","Predictor"))

if add_selectbox == "Introduction of data":
	st.image("Fuel.jpg")
	st.title("Auto-mpg dataset")
	st.header("About Dataset")
	st.image("information.png")
	st.text("Dateset comes from:https://archive.ics.uci.edu/ml/datasets/auto+mpg")
	st.header("Source:")
	st.text("This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the 1983 American Statistical Association Exposition.")
	st.header("What Dataset looks like")
	st.write(mpg_df.head())
	st.header("Attribute Information")
	st.text("1. mpg: continuous.")
	st.text("2. cylinders: multi-valued discrete.")
	st.text("3. displacement: continuous.")
	st.text("4. horsepower: continuous.")
	st.text("5. weight: continuous.")
	st.text("6. acceleration: continuous.")
	st.text("7. model year: multi-valued discrete.")
	st.text("8. origin: multi-valued discrete.")
	st.text("9. car name: string (unique for each instance).")

if add_selectbox == "Check Dataset and EDA":
	st.header("Check Dataset")
	st.write(mpg_df.describe())
	def auto_preprocess(dataframe):
		df_ = dataframe.copy()
		auto_misspelled = {'chevroelt': 'chevrolet',
                       'chevy': 'chevrolet',
                       'vokswagen': 'volkswagen',
                       'vw': 'volkswagen',
                       'hi': 'harvester',
                       'maxda': 'mazda',
                       'toyouta': 'toyota',
                       'mercedes-benz': 'mercedes'}

		df_['brand'] = [auto_misspelled[key].title() if key in auto_misspelled else 
					   key.title() for key in [i.split()[0] for i in df_['car name']]]
		df_['model'] = [' '.join(i.split()[1:]).title() for i in df_['car name']]
		df_ = df_.drop(columns = ['car name'], axis = 1)
		return df_
	df = auto_preprocess(mpg_df)
	st.write(df.head())
	def one_hot_encoder(dataframe, categorical_cols: list, drop_first: bool = False):
		dataframe = pd.get_dummies(dataframe,
                               columns = categorical_cols,
                               drop_first = drop_first)
		return dataframe

	def label_encoder(dataframe, binary_col):
		labelencoder = LabelEncoder()
		dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
		return dataframe

	binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
	print('Binary Features: {}'.format(binary_cols))

	ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
	ohe_cols.append('brand')
	print('Multiclass Features: {}'.format(ohe_cols))
	st.text("First we need check if there any missing data")
	st.write(df.isna().sum())
	st.text("The dataset contains a few unknown values in column horsepower.")
	st.text("Drop those NaN values")
	df = df.dropna()
	st.text("check it again")
	st.write(df.isna().sum())
	st.text("No missing data right now")
	st.title("Feature Engineering")
	sns.set_style('darkgrid')
	plt.figure(figsize = (8, 6))
	sns.distplot(df.mpg, fit= norm)
	plt.show()

	(mu, sigma) = norm.fit(df["mpg"])
	print("mu: {} sigma = {}".format(mu, sigma))
	st.write("mu:",mu,"sigma:",sigma)
	df["mpg"] = np.log1p(df["mpg"])
	(mu, sigma) = norm.fit(df["mpg"])
	st.write("After Transformation,","mu:",mu,"sigma:",sigma)


	st.text("The origin column is really categorical, not numeric. So convert that to a one-hot")
	st.text("The cylinders column is really categorical, So convert that to a one-hot")
	df['cylinders'] = df['cylinders'].astype(int)
	df['origin'] = df['origin'].astype(int)
	df = one_hot_encoder(df, ohe_cols)
	st.text("The new dataset looks like")
	st.write(df.head())
	st.header("EDA")
	st.text("Check horsepower after dropna")
	dis_hor,ax = plt.subplots()
	sns.distplot(df.horsepower)
	st.write(dis_hor)
	MPG_df = plt.figure(figsize=(10,6)); ax = MPG_df.gca()
	df.hist(bins=30, ax=ax)
	plt.suptitle('Auto-mpg', y=1.03)    
	plt.tight_layout()
	st.markdown("Data distribution of each value.")
	st.write(MPG_df)
	st.text("Check corralation for numeric data")
	data_df = list(df.columns[0:7])
	fig_corr,ax = plt.subplots()
	sns.heatmap(df[data_df].corr(), ax=ax, annot = True)
	st.write(fig_corr)

if add_selectbox == "ML step":
	def auto_preprocess(dataframe):
		df_ = dataframe.copy()
		auto_misspelled = {'chevroelt': 'chevrolet',
                       'chevy': 'chevrolet',
                       'vokswagen': 'volkswagen',
                       'vw': 'volkswagen',
                       'hi': 'harvester',
                       'maxda': 'mazda',
                       'toyouta': 'toyota',
                       'mercedes-benz': 'mercedes'}

		df_['brand'] = [auto_misspelled[key].title() if key in auto_misspelled else 
					   key.title() for key in [i.split()[0] for i in df_['car name']]]
		df_['model'] = [' '.join(i.split()[1:]).title() for i in df_['car name']]
		df_ = df_.drop(columns = ['car name'], axis = 1)
		return df_
	df = auto_preprocess(mpg_df)
	def one_hot_encoder(dataframe, categorical_cols: list, drop_first: bool = False):
		dataframe = pd.get_dummies(dataframe,
                               columns = categorical_cols,
                               drop_first = drop_first)
		return dataframe

	def label_encoder(dataframe, binary_col):
		labelencoder = LabelEncoder()
		dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
		return dataframe

	binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
	print('Binary Features: {}'.format(binary_cols))

	ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
	ohe_cols.append('brand')
	print('Multiclass Features: {}'.format(ohe_cols))
	df = df.dropna()

	(mu, sigma) = norm.fit(df["mpg"])
	print("mu: {} sigma = {}".format(mu, sigma))
	df["mpg"] = np.log1p(df["mpg"])
	(mu, sigma) = norm.fit(df["mpg"])
	df['cylinders'] = df['cylinders'].astype(int)
	df['origin'] = df['origin'].astype(int)
	df = one_hot_encoder(df, ohe_cols)
	useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.03).any(axis=None)]

	df.drop(useless_cols, axis = 1, inplace=True)
	st.text("Number of useless variables: 20,Than we drop it")
	code3 = '''	useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.03).any(axis=None)]

df.drop(useless_cols, axis = 1, inplace=True)'''
	st.code(code3, language='python')
	st.text("We got our dataset to do modeling, it looks like")
	st.write(df.head())
	st.write(df.describe())
	st.title("Preprocess Data")
	st.text('Train-Test Split')
	code1 = '''
	X = df.drop(columns = ["mpg","model"])
y = df['mpg']
test_size = 0.2
random_state = 42
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
print(X_train)
print(y_train)
print(X_train Shape: {X_train.shape},X_test Shape: {X_test.shape}')
print(y_train Shape: {y_train.shape},y_test Shape: {y_test.shape}')'''
	st.code(code1, language='python')
	X = df.drop(columns = ["mpg","model"],axis = 1)
	y = df['mpg']
	test_size = 0.2
	random_state = 42
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state = random_state)
	st.write(X_train)
	st.write(y_train)
	st.write("X_train Shape:",X_train.shape,"X_test Shape:",X_test.shape,"y_train Shape:",y_train.shape,"y_test Shape:",y_test.shape)
	st.title("Modeling")

	def train_pipeline(pipe):
		result = dict()
		scaler = pipe.steps[0][1].__class__.__name__
		regressor = pipe.steps[1][1].__class__.__name__
		result['model'] = regressor
		result['scaler'] = scaler if scaler != 'NoneType' else 'Without Scaling'
		pipe.fit(X_train, y_train)
		y_pred = pipe.predict(X_test)
		y_test_exp = np.expm1(y_test)
		y_pred_exp = np.expm1(y_pred)
		result['r2'] =  r2_score(y_test_exp, y_pred_exp),
		result['mse'] =  mean_squared_error(y_test_exp, y_pred_exp),
		result['rmse'] =  mean_squared_error(y_test_exp, y_pred_exp, squared = False)
		result['msle'] =  mean_squared_log_error(y_test_exp, y_pred_exp),
		result['mape'] =  mean_absolute_percentage_error(y_test_exp, y_pred_exp),
		result['mae'] =  mean_absolute_error(y_test_exp, y_pred_exp)
		return result
	scalers = [None, StandardScaler(), RobustScaler(), MinMaxScaler()]

	regressors = [KNeighborsRegressor(), LinearRegression(),
              Lasso(), Ridge(), XGBRegressor(),
              LGBMRegressor(), AdaBoostRegressor(), SVR(),
              RandomForestRegressor(), DecisionTreeRegressor(),
              ExtraTreesRegressor(), GradientBoostingRegressor(),
              CatBoostRegressor(silent = True, allow_writing_files = False)]
    
	eval_data = pd.DataFrame()
	for reg in regressors:
		for sc in scalers:
			pipeline = Pipeline([('scaler', sc), ('reg', reg)])
			eval_data = eval_data.append(pd.DataFrame(train_pipeline(pipeline)))
		eval_data = eval_data.reset_index(drop = True)
	eval_data.sort_values('rmse')
	st.write(eval_data)

if add_selectbox == "Predictor":
	st.title("Predictor")
	st.markdown("As we get from Modeling Step, the prediction will based on the CatBoost")
	st.subheader("user defined prediction")
	displacement = st.slider("Chose The Displacement", value = 195,min_value=68, max_value=455)
	horsepower = st.slider("Chose The Horsepower", value = 105,min_value=46, max_value=230)
	weight = st.slider("Chose The Weight", value = 2977,min_value=1613, max_value=5140)
	acceleration = st.slider("Chose The acceleration", value = 15.5,min_value=8.0, max_value=24.8)
	model = st.slider("Chose The Model year", value = 75,min_value=70, max_value=82)

	if st.button("Predict"):
		list= np.array([displacement,horsepower,weight,acceleration,model])
		predict_df = CatBoostRegressor(pd.DataFrame(list))
	st.write(predict_df)







		








	























     
	






	
	



	

	 			








