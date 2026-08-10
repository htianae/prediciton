from metrisapi.dataanalysis import DataAnalysisClient
from metrisapi.configuration import ConfigurationClient
from metrisapi.helpers import process_trend_values
from metrisapi.historian import (HistorianClient, TrendValuesParameters, InterpolationResolutionType,
                                 AggregateFunction, ticks_from)

import numpy
from matplotlib import pyplot as plt
from IPython.display import display, Markdown

import pandas as pd
import datetime, time
from datetime import datetime
from random import randrange
from datetime import timedelta
import scipy, copy, pickle
import scipy.stats
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from sklearn.svm import OneClassSVM 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

import plotly.express as px
import plotly.graph_objects as go

base_uri = 'https://localhost:9000'
token = 'e'

dac = DataAnalysisClient(base_uri, lambda: token)
hc = HistorianClient(base_uri, lambda: token)
cc = ConfigurationClient(base_uri, token=lambda: token)

def get_data(my_tag_list,my_resolution,start_time,end_time):
    tag_lists = dac.get_tag_lists()
    tag_list_id = [i['id'] for i in tag_lists if i['name']==my_tag_list][0]
    tag_list = dac.get_tag_list(tag_list_id)

    noframe=True
    for tag_info in tag_list['tagInfos']:        
        tvp = TrendValuesParameters(
            tag_id=tag_info['tagID'],
            start=start_time,
            end=end_time,
            interpolation_resolution_type=InterpolationResolutionType.ticks,
            interpolation_resolution=my_resolution,
            aggregate_function = AggregateFunction.last)
        trend_values = hc.get_trend_values(tvp)
        timestamps, values = process_trend_values(trend_values)
        if noframe:
            df = pd.DataFrame(index=timestamps)
            noframe = False
        df[tag_info['name']]=values
    print(f'Data {numpy.shape(df)} loaded.')
    return (df)

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)
if True:
    with open('./Active/ModelSVRWater.pickle', 'rb') as old_file:
        svr_o = pickle.load(old_file)
    with open('./Active/ModelLinWater.pickle', 'rb') as old_file:
        lin_o = pickle.load(old_file)
    with open('./Active/ModelRFMWater.pickle', 'rb') as old_file:
        rfm_o = pickle.load(old_file)
    print('Models loaded.')

tagmap ={'2313.71.1124.FI'    : 'Sootblower_steam', 
         '2313.71.1107.FIZ'   : 'Main_steam_flow',
         '2313.71.2204.FI'    : 'Continous_purge',
         '2313.71.0061.FIC' : 'Feedwater_flow',
         '2313.71.1106.PIZ' : 'Main_steam_pressure',
         '2313.71.0106.LIZ' : 'Drum_level1',
         '2313.71.0107.LIZ' : 'Drum_level2',
         '2313.71.0108.LIZ' : 'Drum_level', #'Drum_level3',
         # ''   : 'Turbopump',
         '2313.71.2003.PI'   : 'Furnace_pressure 1',
         '2313.71.2004.PI'   : 'Furnace_pressure 2',
         '2313.71.2005.PI'   : 'Furnace_pressure 3',
         '2313.71.0946.AI' : 'O2_residual_stack',
         '2313.71.3054.FIZ'   : 'Black_liquor_flow',
         '2313.71.3052.DIZ'    : 'Black_liquor_dry_solids 1',
         '2313.71.3053.DIZ'    : 'Black_liquor_dry_solids 2',
         '2313.71.2017.TI'    : 'Eco2_temperature_left',
         '2313.71.2018.TI'    : 'Eco2_temperature_right',
         '2313.71.6005.QI'    : 'Saturated_steam_conductivity',
         '2313.71.6006.QI'    : 'Main_steam_conductivity',
            '2313.71.2222.FICZ': 'Primary_air_flow',
         # '': 'Secondary_air_flow',
         '2313.71.2228.FIC' : 'Tertiary_air_flow A',
         # '' : 'Tertiary_air_flow B',
         '2313.71.2046.SIC' : 'ID_fan 1',
         '2313.71.2060.SIC' : 'ID_fan 2',
         '2313.71.2304.FIZ': 'DNCG_flow',
         '2313.71.2315.FIZ': 'DTVG_1 flow',
         '2313.71.1122.PIC': "Soot_valve",
         '2313.71.8201.QI': "RecBo_condensate_conductivity",
         }

tags=['Main_steam_flow',
      'Sootblower_steam',
      'Continous_purge',
      'Feedwater_flow',
      'Main_steam_pressure',
      'Drum_level',
      # 'Turbopump', # tag not existing
      'Soot_valve',
      # 'Feed_turbo_pump' # tag not existing
     ]
# target_name='Main_steam_flow'
target_name='Feedwater_flow'

r2delta = 0.01                                              # how much the r2 must improve
Sampling = ticks_from(hours=0, minutes=1, seconds=0)        # sampling rate of data, not more than 50k of rows
Full_Test_Period = pd.DateOffset(days = 3)               # Lenght of test data period in days
#d1 = pd.Timestamp('2021-10-20 00:00',tz ='UTC')                        # Start of data
d1 = pd.Timestamp('2025-11-18 00:00',tz ='UTC')                        # Start of data
# d1 = datetime.now()-pd.DateOffset(months = 0, days = 18+7)                        # Start of data
d2 = random_date(datetime.now()-pd.DateOffset(hours = 2*24)   ,datetime.now()-pd.DateOffset(hours = 1*24) )
# d1 = d1.tz_localize('UTC')
d2 = d2.tz_localize('UTC')

Data_Period_1 = [d1, d2] 
print(f'Recent training data loaded for period with {Sampling/1e7:.0f} s sampling :')
print(Data_Period_1)
Raw_WLA = get_data('WLA tags',Sampling,Data_Period_1[0],Data_Period_1[1])
#     Raw_WLA = pd.concat([Raw_WLA_0,Raw_WLA_1],copy=True)

#%%timeit
for retrain_loop in range(0,1):
    # Initialize an empty dataframe. Fill with dataframe data and generic columns names.
    WLA=pd.DataFrame([],columns=tags)
    for ct in tags:   
        for tmk,tmv in zip(tagmap.keys(),tagmap.values()):
            if tmv == ct:
                WLA[ct] = Raw_WLA[tmk]
    
    # clean and filter data. Define train and test data sets.
    WLA = WLA.dropna()
     
    WLA['filter'] =WLA['Main_steam_flow'].rolling('2h').mean().rolling('12h',center=True).min()
    WLA = WLA[WLA['filter']>50]
    WLA = WLA.drop(columns=['filter',])
    
    print (f'Final datasize {numpy.shape(WLA)}')
    WLAs = WLA.copy()
    WLAtrain   = WLAs[ WLAs.index < WLAs.index[-1] - Full_Test_Period ]
    WLAtest    = WLAs[ WLAs.index > WLAtrain.index[-1] ]
    
    # set pipelines for scaling and regression.
    pipe1  = Pipeline([('scaler',StandardScaler()),('SVR',svm.SVR(C=50,cache_size=2000,epsilon=4) )])
    pipe2  = Pipeline([('scaler',StandardScaler()),('Lin',linear_model.HuberRegressor() )]) 
    pipe3  = Pipeline([('scaler',StandardScaler()),('RFM',RandomForestRegressor(max_depth=6, 
                                                                                max_features='sqrt',
                                                                                min_samples_leaf=5,n_estimators = 125) )])
    
    # defines train  and test targets
    target_train = WLAtrain[target_name] 
    target_test  = WLAtest[ target_name] 
    
    print (f'Model training SVR')
    # fit all three models for training data
    pipe1.fit(WLAtrain.drop(columns=[target_name]), target_train)
    print (f'Model training Lin')
    pipe2.fit(WLAtrain.drop(columns=[target_name]), target_train)
    print (f'Model training RFM')
    pipe3.fit(WLAtrain.drop(columns=[target_name]), target_train)
##################################################################################################################
    # save the models to retraining folder
    filedate = datetime.now().strftime(' %Y-%b-%d %Hh %Mm')
    pickle_files = [f'./Retrain/ModelSVRWater{filedate:s}L{len(WLAtrain):d}.pickle',
                    f'./Retrain/ModelLinWater{filedate:s}L{len(WLAtrain):d}.pickle',
                    f'./Retrain/ModelRFMWater{filedate:s}L{len(WLAtrain):d}.pickle']
    models = [pipe1,pipe2,pipe3]
   
    for pickle_file,pipe in zip(pickle_files,models):
        with open(pickle_file, 'wb') as file1:
            pickle.dump(pipe, file1, pickle.HIGHEST_PROTOCOL)
    print('Models saved to retraining folder.')
##########################################################################################################################
    print (f'Model predicting')
    # use new pipe1,pipe2, and pipe3 models to predict on test data
    pred_test1 = pipe1.predict( WLAtest.drop(columns=[target_name]).copy() )
    pred_test2 = pipe2.predict( WLAtest.drop(columns=[target_name]).copy() )
    pred_test3 = pipe3.predict( WLAtest.drop(columns=[target_name]).copy() )
    # use active svr_o,lin_o, and frm_o models to predict on test data
    pred_test1_o = svr_o.predict( WLAtest.drop(columns=[target_name]).copy() )
    pred_test2_o = lin_o.predict( WLAtest.drop(columns=[target_name]).copy() )
    pred_test3_o = rfm_o.predict( WLAtest.drop(columns=[target_name]).copy() )

    # insert predictions to the dataframe of test data
    WLAtest.insert(len(WLAtest.columns),'Target',    target_test)
    WLAtest.insert(len(WLAtest.columns),'Prediction SVM',pred_test1)
    WLAtest.insert(len(WLAtest.columns),'Prediction Lin',pred_test2)
    WLAtest.insert(len(WLAtest.columns),'Prediction RFM',pred_test3)
    WLAtest.insert(len(WLAtest.columns),'Prediction SVM orig.',pred_test1_o)
    WLAtest.insert(len(WLAtest.columns),'Prediction Lin orig.',pred_test2_o)
    WLAtest.insert(len(WLAtest.columns),'Prediction RFM orig.',pred_test3_o)
    
    # calculate r2 score for new and active (_o postfix) models
    r2n_1 = r2_score(WLAtest['Target'],WLAtest['Prediction SVM'])
    r2n_2 = r2_score(WLAtest['Target'],WLAtest['Prediction Lin'])
    r2n_3 = r2_score(WLAtest['Target'],WLAtest['Prediction RFM'])
    r2n_1_o = r2_score(WLAtest['Target'],WLAtest['Prediction SVM orig.'])
    r2n_2_o = r2_score(WLAtest['Target'],WLAtest['Prediction Lin orig.'])
    r2n_3_o = r2_score(WLAtest['Target'],WLAtest['Prediction RFM orig.'])
    
    # # Save first models
    # with open('./Active/ModelSVRWater.pickle','wb') as file1:
    #         pickle.dump(pipe1, file1, pickle.HIGHEST_PROTOCOL)
    # with open('./Active/ModelLinWater.pickle','wb') as file1:
    #     pickle.dump(pipe2, file1, pickle.HIGHEST_PROTOCOL)
    # with open('./Active/ModelRFMWater.pickle','wb') as file1:
    #     pickle.dump(pipe3, file1, pickle.HIGHEST_PROTOCOL)
    
    # compare new models to active and  replace if an improvement is found
    # comment out or remove the lines 84,89,94 before running this for testing purposes
    New_good = 0
    if r2n_1 > r2n_1_o + r2delta: #--0.2 was here may 28th 
        print(f'SVM R2 [new {r2n_1:5.2f}] > [active {r2n_1_o:5.2f}]. New model saved as active.')
        with open('./Active/ModelSVRWater.pickle','wb') as file1:
                pickle.dump(pipe1, file1, pickle.HIGHEST_PROTOCOL)
        New_good = 1
    if r2n_2 > r2n_2_o + r2delta:
        print(f'Lin R2 [new {r2n_2:5.2f}] > [active {r2n_2_o:5.2f}]. New model saved as active.')
        with open('./Active/ModelLinWater.pickle','wb') as file1:
            pickle.dump(pipe2, file1, pickle.HIGHEST_PROTOCOL)
        New_good = New_good + 1
    if r2n_3 > r2n_3_o + r2delta:
        print(f'RFM R2 [new {r2n_3:5.2f}] > [active {r2n_3_o:5.2f}]. New model saved as active.')
        with open('./Active/ModelRFMWater.pickle','wb') as file1:
            pickle.dump(pipe3, file1, pickle.HIGHEST_PROTOCOL)
        New_good = New_good +1
    print(f'Improvement found for {New_good:d} models.')

    print (f'Plot trend')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction SVM'],
                    mode='markers', name=f'Prediction SVM R2:{r2n_1:5.2f}'))
    fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction Lin'],
                    mode='markers', name=f'Prediction Lin R2:{r2n_2:5.2f}'))
    fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction RFM'],
                    mode='markers', name=f'Prediction RFM R2:{r2n_3:5.2f}')) 
    fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Target'],
                    mode='lines', name=f'Measurement'))
 
    fig.update_layout(
        title=f'{target_name:s} trends ', xaxis_title='Date time', yaxis_title=f'{target_name:s}',
        legend_title="Legend:", font=dict(size=14,color="Black"),
        annotations=[ dict(  x=0.7, y=0.1, text="(c) ANDRITZ Metris",showarrow=False, xref="paper", yref="paper")] )        
    fig.show()
print('Ready.')

fig = go.Figure()
fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction SVM'],
                mode='markers', name=f'Prediction SVM R2:{r2n_1:5.2f}'))
fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction SVM orig.'],
                mode='markers', name=f'Prediction SVM orig. R2:{r2n_1_o:5.2f}'))
fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction Lin'],
                mode='markers', name=f'Prediction Lin R2:{r2n_2:5.2f}'))
fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction Lin orig.'],
                mode='markers', name=f'Prediction Lin orig. R2:{r2n_2_o:5.2f}'))
fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction RFM'],
                mode='markers', name=f'Prediction RFM R2:{r2n_3:5.2f}'))
fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Prediction RFM orig.'],
                mode='markers', name=f'Prediction RFM orig. R2:{r2n_3_o:5.2f}'))
fig.add_trace(go.Scatter(x=WLAtest.index, y=WLAtest['Target'],
                mode='lines', name=f'Measurement'))


fig.update_layout(
    title=f'{target_name:s} trends ', xaxis_title='Date time', yaxis_title=f'{target_name:s}',
    legend_title="Legend:", font=dict(size=14,color="Black"),
    annotations=[ dict(  x=0.7, y=0.1, text="(c) ANDRITZ Metris",showarrow=False, xref="paper", yref="paper")] )        
fig.show()