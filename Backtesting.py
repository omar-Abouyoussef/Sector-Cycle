import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
from datetime import datetime as dt
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from retry import retry
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Utils.Sector_Cycle import main
import pyfolio as pf
import plotly.tools as tls
import streamlit as st



@retry((Exception), tries=10, delay=1, backoff=0)
def getUSETFdata(etf:str,n:int,freq:str,date):
    """_summary_

    Args:
        etf (str): etf name
        n (int): last n bars
        freq (str): ["Daily", "Weekly", "Monthly"]
        date (datetime object):date used for caching purposes

    Returns:
        pandas dateframe: close price
    """

    interval_dic = {'Daily':Interval.in_daily, 'Weekly':Interval.in_weekly, 'Monthly':Interval.in_monthly}
    tv = TvDatafeed()
    response = tv.get_hist(symbol=f'{etf}',
                    exchange='AMEX',
                    interval=interval_dic[freq],
                    n_bars=n)['close']
    response = pd.DataFrame(response)
    return response



date = dt.today().date()
sector_symbol = st.selectbox(label='Sector:',
                     options=['MM', 'SB', 'SE', 'SF', 'SI', 'SK', 'SL', 'SP', 'SS', 'SU', 'SV', 'SY'],
                     key='sector_symbol'
                     )

sector_etf = {'MM':'SPY','SB':'XLB','SE':'XLE',
              'SF':'XLF','SI':'XLI','SK':'XLK',
              'SL':'XLC','SP':'XLP','SS':'XLRE',
              'SU':'XLU', 'SV':'XLV', 'SY':'XLY'}

fig, factor = main(sector_symbol)
etf = getUSETFdata(etf=sector_etf[sector_symbol],n=2000,freq='Daily',date=date)
etf.name = sector_etf[sector_symbol]

etf.index = pd.to_datetime(etf.index.date)
factor.index = pd.to_datetime(factor.index.date)
etf.to_csv('etf.csv')
factor.to_csv('factor.csv')



class FactorStrategy(bt.Strategy):
    # slope_threshold can be adjusted to ignore very small changes

    def __init__(self):
        self.etf = self.datas[0]
        self.factor = self.datas[1]
        # No indicator needed; we'll compute slope manually

        # Record trade signals for plotting later
        self.trade_signals = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0)
            if order.isbuy():
                self.trade_signals.append({'date': dt, 'price': order.executed.price, 'signal': 'buy'})
                self.log(f'BUY EXECUTED at {order.executed.price:.2f}')
            elif order.issell():
                self.trade_signals.append({'date': dt, 'price': order.executed.price, 'signal': 'sell'})
                self.log(f'SELL EXECUTED at {order.executed.price:.2f}')

    def next(self):
        # Ensure we have at least two bars to compute a difference
        if len(self) < 2:
            return

        etf_pos = self.getposition(self.etf).size
        # Compute the slope as the difference between current and previous factor close
        slope = self.factor.close[0] - self.factor.close[-1]
        self.log(f"Factor slope: {slope:.4f}")

        # If the slope is positive (upward trending) beyond a threshold, buy if not in position.
        if slope > 0:
            if etf_pos == 0:
                self.buy(data=self.etf)
                self.log(f"BUY ORDER PLACED: ETF close: {self.etf.close[0]:.2f}, slope: {slope:.4f}")
        # If the slope is negative (downward trending) beyond the threshold, exit if in position.
        elif slope < 0:
            if etf_pos > 0:
                self.close(data=self.etf)
                self.log(f"SELL ORDER PLACED: ETF close: {self.etf.close[0]:.2f}, slope: {slope:.4f}")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")


# -----------------------
# Data Feeds Setup
# -----------------------
# For both CSV files, we assume:
#  - Column 0: Date (formatted as YYYY-MM-DD)
#  - Column 1: Close Price
# For ETF, we simulate open/high/low using the close column.
etfdata = btfeeds.GenericCSVData(
    dataname='etf.csv',
    datetime=0,
    dtformat=('%Y-%m-%d'),
    open=1,
    high=None,
    low=None,
    close=1,
    volume=None,
    openinterest=None,
    timeframe=bt.TimeFrame.Days,
    name='ETF'
)

factordata = btfeeds.GenericCSVData(
    dataname='factor.csv',
    datetime=0,
    dtformat=('%Y-%m-%d'),
    open=1,
    high=None,
    low=None,
    close=1,
    volume=None,
    openinterest=None,
    timeframe=bt.TimeFrame.Days,
    name='Factor'
)


# -----------------------
# Cerebro Setup and Run
# -----------------------
cerebro = bt.Cerebro()
# Order matters: ETF is datas[0] and Factor is datas[1]
cerebro.adddata(etfdata)
cerebro.adddata(factordata)
cerebro.addstrategy(FactorStrategy)
cerebro.broker.setcash(100.0)
cerebro.addanalyzer(bt.analyzers.PyFolio)


print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Retrieve trade signals from the strategy instance
strategy_instance = results[0]
trade_signals = strategy_instance.trade_signals
pyfolio = results[0].analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfolio.get_pf_items()
st.session_state.returns = returns
st.session_state.positions = positions
st.session_state.transactions = transactions
st.session_state.gross_lev = gross_lev


# -----------------------
# Load Full Data with Pandas for Plotly
# -----------------------
# -----------------------
# Load Full Data with Pandas for Plotly
# -----------------------
etf_df = pd.read_csv('etf.csv', parse_dates=[0])
etf_df.columns = ['date', 'close']

factor_df = pd.read_csv('factor.csv', parse_dates=[0])
factor_df.columns = ['date', 'Factor']

# -----------------------
# Create Interactive Plotly Chart with Subplots
# -----------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.7, 0.3],
    subplot_titles=("ETF Price with Trade Signals", "Factor")
)

# Top subplot: ETF Price
fig.add_trace(go.Scatter(
    x=etf_df['date'],
    y=etf_df['close'],
    mode='lines',
    name='ETF Price',
    line=dict(color='black', width=1)

), row=1, col=1)

# Add trade signals to the ETF chart
buy_signals = [sig for sig in trade_signals if sig['signal'] == 'buy']
sell_signals = [sig for sig in trade_signals if sig['signal'] == 'sell']

if buy_signals:
    buy_df = pd.DataFrame(buy_signals)
    fig.add_trace(go.Scatter(
        x=buy_df['date'],
        y=buy_df['price'],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Buy Signal'
    ), row=1, col=1)

if sell_signals:
    sell_df = pd.DataFrame(sell_signals)
    fig.add_trace(go.Scatter(
        x=sell_df['date'],
        y=sell_df['price'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Sell Signal'
    ), row=1, col=1)

# Bottom subplot: Factor series
fig.add_trace(go.Scatter(
    x=factor_df['date'],
    y=factor_df['Factor'],
    mode='lines',
    name='Factor',
    line=dict(color='black', width=1)

), row=2, col=1)

# Update layout for interactivity and proper date display
fig.update_layout(
    title="Interactive Chart with ETF Price, Trade Signals, and Factor",
    xaxis=dict(type="date"),
    xaxis2=dict(type="date", title="Date"),
    yaxis_title="ETF Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified', hoversubplots="single"
)

fig.add_hline(y=-1.27, row=2,col=1)
fig.add_hline(y=1.27, row=2,col=1)

st.plotly_chart(fig)


pf.create_returns_tear_sheet(
    returns=st.session_state.returns,
    transactions=st.session_state.transactions,
    live_start_date='2022-01-01',
    estimate_intraday=False,
    round_trips=False)
