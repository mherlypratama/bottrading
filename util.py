import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import tree
import pickle


def get_ema(ticker_data, period):
    ema = ticker_data['Close'].ewm(span=period).mean()
    column_name = 'ema_' + str(period)
    ticker_data[column_name] = ema
    return ticker_data


def get_ticker_data(ticker_symbol, data_period, data_interval):
    ticker_data = yf.download(tickers=ticker_symbol,
                              period=data_period, interval=data_interval)
    if len(ticker_data) == 0:
        st.write(
            'Could not find the ticker data. Modify ticker symbol or reduce the Period value.')
    else:
        # Format the x-axis to skip dates with missing values
        ticker_data.index = ticker_data.index.strftime("%d-%m-%Y %H:%M")
    return ticker_data


def get_candle_chart(candle_fig, ticker_data):
    candle_fig.add_trace(
        go.Candlestick(x=ticker_data.index,
                       open=ticker_data['Open'],
                       close=ticker_data['Close'],
                       low=ticker_data['Low'],
                       high=ticker_data['High'],
                       name='Market Data'
                       )
    )
    candle_fig.update_layout(
        height=800,
    )
    return candle_fig


def add_ema_trace(candle_fig, timestamp, ema, trace_name, color):
    candle_fig.add_trace(
        go.Scatter(
            x=timestamp,
            y=ema,
            name=trace_name,
            line=dict(color=color)
        )
    )
    return candle_fig


def add_trades_trace(candle_fig, ticker_data):
    candle_fig.add_trace(
        go.Scatter(
            x=ticker_data.index,
            y=ticker_data['Trade Price'],
            name='Trade Triggers',
            marker_color=ticker_data['Trade Color'],
            mode='markers'
        )
    )
    return candle_fig


def add_row_trace(candle_fig, x_value, y_value, trace_name, color, row_num, mode='lines'):
    candle_fig.add_trace(
        go.Scatter(
            x=x_value,
            y=y_value,
            name=trace_name,
            line=dict(color=color),
            mode=mode
        ),
        row=row_num,
        col=1
    )
    return candle_fig


def create_ema_trade_list(ticker_data, ema1_col_name, ema2_col_name):
    ticker_data['ema_diff'] = ticker_data[ema1_col_name] - \
        ticker_data[ema2_col_name]
    prev_state = 'unknown'
    trades = []
    for i in range(len(ticker_data)):
        if ticker_data['ema_diff'][i] >= 0:
            state = 'positive'
        else:
            state = 'negative'
        if prev_state != 'unknown':
            if state == 'positive' and prev_state == 'negative':
                try:
                    trade = str(
                        ticker_data.index[i+1]) + ',' + str(ticker_data['Open'][i+1]) + ',buy,cyan'
                    trade = trade.split(',')
                    trades.append(trade)
                except:
                    continue
            elif state == 'negative' and prev_state == 'positive':
                try:
                    trade = str(
                        ticker_data.index[i+1]) + ',' + str(ticker_data['Open'][i+1]) + ',sell,magenta'
                    trade = trade.split(',')
                    trades.append(trade)
                except:
                    continue
        prev_state = state
    return trades


def create_sar_trade_list(ticker_data, rr, atr_mult):
    trades = []
    state = 'neutral'
    for i in range(len(ticker_data)):
        if ticker_data['Close'][i] > ticker_data['sar'][i] and state != 'buy':
            try:
                buy_price = ticker_data['Open'][i+1]
                sl = ticker_data['Open'][i+1] - \
                    ticker_data['atr'][i] * float(atr_mult)
                tp = ticker_data['Open'][i+1] + \
                    float(rr) * ticker_data['atr'][i] * float(atr_mult)
                trade = str(ticker_data.index[i+1]) + ',' + str(
                    buy_price) + ',' + str(sl) + ',' + str(tp) + ',long,cyan'
                trade = trade.split(',')
                trades.append(trade)
                state = 'buy'
            except:
                continue
        elif ticker_data['Close'][i] < ticker_data['sar'][i] and state != 'sell':
            try:
                buy_price = str(ticker_data['Open'][i+1])
                sl = ticker_data['Open'][i+1] + \
                    ticker_data['atr'][i] * float(atr_mult)
                tp = ticker_data['Open'][i+1] - \
                    float(rr) * ticker_data['atr'][i] * float(atr_mult)
                trade = str(ticker_data.index[i+1]) + ',' + str(
                    buy_price) + ',' + str(sl) + ',' + str(tp) + ',short,magenta'
                trade = trade.split(',')
                trades.append(trade)
                state = 'sell'
            except:
                continue
    return trades


def join_trades_to_ticker_data(trades, ticker_data):
    trades_df = convert_trades_to_df(trades)
    trades_df['start_pos_price'] = trades_df['start_pos_price'].astype(
        float).round(4)
    ticker_data = pd.concat([ticker_data, trades_df], axis=1, join='outer')
    return ticker_data


def convert_trades_to_df(trades):
    return pd.DataFrame(trades, columns=['time', 'start_pos_price', 'sl', 'tp', 'trade_type', 'trade_color', 'p/l', 'status']).set_index('time')


def simulate_ema_cross_trading(trades):
    results = []
    buy_price = 0
    for trade in trades:
        if trade[3] == 'buy':
            buy_price = float(trade[1])
        elif buy_price != 0:
            sell_price = float(trade[1])
            results.append(sell_price - buy_price)
    return results


def get_sim_summary(trade_pl, share_amount, initial_capital):
    accumulative_account_value = []
    np_sim_results = np.array(trade_pl)
    win_rate = (np_sim_results > 0).sum() / len(trade_pl) * 100
    win_rate = win_rate.round(1)
    sim_results_df = pd.DataFrame(trade_pl, columns=['Change'])
    sim_fig = go.Figure()
    sim_fig.add_trace(
        go.Scatter(
            x=sim_results_df.index,
            y=sim_results_df['Change']
        )
    )
    accumulative_account_value.append(initial_capital)
    total = float(initial_capital)
    for item in trade_pl:
        price_change = float(item)
        share_amount = float(share_amount)
        total = total + price_change*share_amount
        accumulative_account_value.append(total)
    accumulative_account_value_df = pd.DataFrame(
        accumulative_account_value, columns=['Acc Value'])
    accumulative_fig = go.Figure()
    accumulative_fig.add_trace(
        go.Scatter(
            x=accumulative_account_value_df.index,
            y=accumulative_account_value_df['Acc Value']
        )
    )
    return win_rate, sim_results_df, sim_fig, accumulative_fig


def display_sim_results(win_rate, trades_df, sim_fig, accumulative_fig):
    st.write('Win Rate:', win_rate, '%')
    st.write(trades_df['p/l'].describe())
    st.write(trades_df)
    st.write(accumulative_fig)


def simulate_trades(trades, ticker_data):
    results = []
    for i in range(len(trades)):
        trade_row = trades[i]
        time = trade_row[0]
        start_pos_price = float(trade_row[1])
        stop_loss = float(trade_row[2])
        take_profit = float(trade_row[3])
        trade_type = trade_row[4]
        if 'nan' not in trade_row:
            start_pos_index = ticker_data.index.get_loc(time)
            for j in range(start_pos_index, len(ticker_data)):
                try:
                    close_price = ticker_data.iloc[j]['Close']
                    low_price = ticker_data.iloc[j]['Low']
                    high_price = ticker_data.iloc[j]['High']
                except:
                    break
                if trade_type == 'long':
                    if high_price >= take_profit:
                        trades[i].append(high_price - start_pos_price)
                        trades[i].append(1)
                        break
                    elif low_price <= stop_loss:
                        trades[i].append(low_price - start_pos_price)
                        trades[i].append(0)
                        break
                elif trade_type == 'short':
                    if low_price <= take_profit:
                        trades[i].append(start_pos_price - low_price)
                        trades[i].append(1)
                        break
                    elif high_price >= stop_loss:
                        trades[i].append(start_pos_price - high_price)
                        trades[i].append(0)
                        break
    return trades


def train_ml_model(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    pickle.dump(clf, open("decision_tree_model.p", "wb"))


def predict_using_saved_model(ticker_data, features, pickle_filename):
    clf = pickle.load(open(pickle_filename, "rb"))
    prediction_result = clf.predict(features)
    ticker_data['prediction'] = prediction_result
    return ticker_data


def create_trade_list_from_prediction(ticker_data, rr, atr_mult):
    trades = []
    recommended_trades_df = ticker_data.loc[ticker_data['prediction'] == 1]
    for i in range(len(recommended_trades_df)):
        buy_price = recommended_trades_df['Open'][i]
        if buy_price > recommended_trades_df['sar'][i]:
            sl = buy_price - recommended_trades_df['atr'][i] * float(atr_mult)
            tp = buy_price + \
                recommended_trades_df['atr'][i] * float(atr_mult) * float(rr)
            trade = str(recommended_trades_df.index[i]) + ',' + str(
                buy_price) + ',' + str(sl) + ',' + str(tp) + ',long,cyan'
        else:
            sl = buy_price + recommended_trades_df['atr'][i] * float(atr_mult)
            tp = buy_price - \
                recommended_trades_df['atr'][i] * float(atr_mult) * float(rr)
            trade = str(recommended_trades_df.index[i]) + ',' + str(
                buy_price) + ',' + str(sl) + ',' + str(tp) + ',short,magenta'
        trade = trade.split(',')
        trades.append(trade)
    return trades
