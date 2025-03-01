from datetime import datetime as dt
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from retry import retry
from factor_analyzer import FactorAnalyzer, ConfirmatoryFactorAnalyzer, ModelSpecificationParser
from scipy.signal import savgol_filter 
from scipy.stats import zscore
import streamlit as st



@retry((Exception), tries=10, delay=1, backoff=0)
@st.cache_data
def get_index_data(sector, suffix,n,freq,date):
    tv = TvDatafeed()
    response = tv.get_hist(symbol=f'{sector}{suffix}',
                    exchange='INDEX',interval=freq,
                    n_bars=n)['close']
    return response

def filter(x, window, order):
    smoothed = savgol_filter(x,window_length=window,polyorder=order)
    smoothed = savgol_filter(smoothed,window_length=10,polyorder=1)
    smoothed = savgol_filter(smoothed,window_length=10,polyorder=1)
    smoothed = savgol_filter(smoothed,window_length=10,polyorder=1)

    return smoothed


def get_data(sector_name,date):
    ma5 = get_index_data(sector_name,'FD',n=2000,freq=Interval.in_daily,date=date)
    ma20 = get_index_data(sector_name,'TW',n=2000,freq=Interval.in_daily,date=date)
    ma50 = get_index_data(sector_name,'FI',n=2000,freq=Interval.in_daily,date=date)
    ma100 = get_index_data(sector_name,'OH',n=2000,freq=Interval.in_daily,date=date)
    ma200 = get_index_data(sector_name,'TH',n=2000,freq=Interval.in_daily,date=date)

    return ma5, ma20, ma50, ma100, ma200


def preprocessing(ma5, ma20, ma50, ma100, ma200, filter_window=30,filter_polyorder=3):
    ma5.name='ma5'
    ma5 = pd.DataFrame(ma5)

    ma5['ma20'] = ma20
    ma5['ma50'] = ma50
    ma5['ma100'] = ma100
    ma5['ma200'] = ma200

    df = ma5.copy()

    df_smoothed = df.apply(filter,args=(filter_window, filter_polyorder))
    return df_smoothed

def factor_model(df):
    
    cfa = FactorAnalyzer(1, rotation = None, method='minres').fit(df.values)
    factors = pd.DataFrame(cfa.transform(df),
                       index = df.index)
    
    return factors

def plot(factors):
    
    series = zscore(factors*-1).sum(axis=1)

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=factors.index, y=series,mode='lines',line=dict(color='black', width=1)))
    
    fig.update_layout(
    title=dict(text="Sector Cycle"))

    fig.add_hline(y=1.27)
    fig.add_hline(y=-1.27)


    return fig

def main(sector_name):
    date = dt.today().date()
    ma5, ma20, ma50, ma100, ma200 = get_data(sector_name=sector_name,date=date)
    df_smoothed = preprocessing(ma5, ma20, ma50, ma100, ma200,filter_window=30,filter_polyorder=3)
    factors = factor_model(df=df_smoothed)
    fig = plot(factors=factors)
    return fig, factors


class App():
    def __init__(self):
        pass

    def run(self):
        
        st.set_page_config(layout="wide", page_title='Sector Cycle Experimental')
        sector_symbol = st.selectbox(label='Sector:',
                     options=['MM', 'SB', 'SE', 'SF', 'SI', 'SK', 'SL', 'SP', 'SS', 'SU', 'SV', 'SY'],
                     key='sector_symbol'
                     )

        self.sector = sector_symbol
        fig, factors = main(self.sector)
        st.plotly_chart(fig)
    
app = App()
app.run()
