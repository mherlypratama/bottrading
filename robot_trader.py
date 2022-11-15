import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import talib
import util


def main():
    ticker_data = util.get_ticker_data(
        ticker_symbol, data_period, data_interval)

    if len(ticker_data) != 0:
        ticker_data['sar'] = talib.SAR(
            ticker_data['High'], ticker_data['Low'], acceleration=0.02, maximum=0.2)
        ticker_data['atr'] = talib.ATR(
            ticker_data['High'], ticker_data['Low'], ticker_data['Close'], timeperiod=14)
        ticker_data.dropna(inplace=True)

        trades = util.create_sar_trade_list(ticker_data, rr, atr_mult)
        # simulasi apakah data trade lost atau rugi
        trades = util.simulate_trades(trades, ticker_data)
        trades_df = util.convert_trades_to_df(trades)  # convert to 1 0
        win_rate, sim_results_df, sim_fig, accumulative_fig = util.get_sim_summary(
            trades_df['p/l'].tolist(), share_amount, initial_capital)
        st.subheader('Parabolic SAR Strategy')  # mengukur winrate
        # tanpa machine larning
        util.display_sim_results(
            win_rate, trades_df, sim_fig, accumulative_fig)

# machine learning train data
        ticker_data = util.join_trades_to_ticker_data(trades, ticker_data)
        ticker_data.fillna({'status': 0}, inplace=True)

        # selisih value sar ke baris sebelumnya
        ticker_data['sar_diff'] = ticker_data['sar'].diff()
        ticker_data['sar_change_rate'] = ticker_data['sar_diff'] / \
            ticker_data['sar'].shift(
                periods=1)  # membagi value sar ke value sar sebelumnya karena berada pada range yang sama
        ticker_data.dropna(
            subset=['sar_diff', 'sar_change_rate'], inplace=True)

        # latih robot

        X = ticker_data[['sar_diff', 'sar_change_rate']].to_numpy()
        Y = ticker_data['status']

        if is_training_mode:
            util.train_ml_model(X, Y)
        else:
            ticker_data = util.predict_using_saved_model(
                ticker_data, X, "decision_tree_model.p")
            simulated_trades = util.create_trade_list_from_prediction(
                ticker_data, rr, atr_mult)
            simulated_trades = util.simulate_trades(
                simulated_trades, ticker_data)
            simulated_trades_df = util.convert_trades_to_df(simulated_trades)
            win_rate, sim_results_df, sim_fig, accumulative_fig = util.get_sim_summary(
                simulated_trades_df['p/l'].tolist(), share_amount, initial_capital)
            st.subheader('With Machine Learning')
            util.display_sim_results(
                win_rate, simulated_trades_df, sim_fig, accumulative_fig)


if __name__ == '__main__':
    ticker_symbol = st.sidebar.text_input(
        "Please enter the stock symbol", 'MSFT'
    )
    is_training_mode = st.sidebar.checkbox('Training Mode', value=True)
    data_period = st.sidebar.text_input('Period', '10d')
    data_interval = st.sidebar.radio('Interval', ['15m', '30m', '1h', '1d'])
    rr = st.sidebar.text_input('Risk/Reward', 2)
    atr_mult = st.sidebar.text_input('ATR Multiple SL', 3)  # aVARAGE TRUE RAGE
    share_amount = st.sidebar.text_input('Number of Shares', 1)
    initial_capital = st.sidebar.text_input('Initial Capital (USD)', 10000)

    st.header("Robot Trader :rocket:")
    st.write("*Warning: not financial advice!* :sunglasses:")
    st.write("---")

    main()
