from backtester.strategy import MACrossover
from apps.data_visualizer.data import Data
from backtester.engine import Engine
from backtester.data import OHLC
from datetime import datetime

import plotly.graph_objs as go

import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Data parameters
DATA_PARAMS = {
    'data_path': (
        r'C:\Users\Gui\PycharmProjects\PyQuant_v1\example_scripts\brazilian_corn_futures\csv_data\ccm_60min.csv'
    ),
    'symbol': 'ccm',
    'timeframe': '60min',
    'date_from': datetime(2020, 1, 1),
    'date_to': datetime.today(),
    'point_value': 450.0,
    'cost_per_trade': 2.50,
}

if __name__ == '__main__':
    import time as tm

    # Start the timer
    start_time = tm.perf_counter()

    # Create OHLC (Open, High, Low, Close) data object
    data = OHLC(symbol=DATA_PARAMS['symbol'], timeframe=DATA_PARAMS['timeframe'])

    # Load data from CSV file
    data.load_data_from_csv(
        filepath=DATA_PARAMS['data_path'], date_from=DATA_PARAMS['date_from'], date_to=DATA_PARAMS['date_to'],
    )

    # Create trading strategy using Moving Average Crossover
    strategy = MACrossover(
        short_ma_func='sma',
        long_ma_func='ema',
        short_ma_period=1,
        long_ma_period=19,
        delta_quantile=0.8,
        delta_window=50,
    )

    # Create backtesting engine
    engine = Engine(point_value=DATA_PARAMS['point_value'], cost_per_trade=DATA_PARAMS['cost_per_trade'])
    result_data = engine.run_backtest(ohlc=data, strategy=strategy, bypass_first_exit_check=True)
    processed_data, registry = result_data['data'], result_data['registry']
    result = registry.get_result(verbose=True)
    trades = registry.trades

    # Create traces for visualization
    candle_data = processed_data.data
    indicator_traces = []
    indicator_traces.append(go.Scatter(
        x=candle_data.index,
        y=candle_data['short_ma'],
        mode='lines',
        line=dict(color='purple', width=2),
        name='Short MA',
    ))
    indicator_traces.append(go.Scatter(
        x=candle_data.index,
        y=candle_data['long_ma'],
        mode='lines',
        line=dict(color='yellow', width=2),
        name='Long MA',
    ))
    subplot_traces = []
    subplot_traces.append(go.Scatter(
        x=candle_data.index,
        y=candle_data['delta'],
        mode='lines',
        line=dict(color='blue', width=1),
        name='Delta',
    ))
    subplot_traces.append(go.Scatter(
        x=candle_data.index,
        y=candle_data['delta_threshold'],
        mode='lines',
        line=dict(color='yellow', width=1, dash='dash'),
        name='Upper Delta Threshold',
    ))
    subplot_traces.append(go.Scatter(
        x=candle_data.index,
        y=-candle_data['delta_threshold'],
        mode='lines',
        line=dict(color='yellow', width=1, dash='dash'),
        name='Lower Delta Threshold',
    ))

    # Save data to app
    appdata = Data(
        ohlc=candle_data, trade_registry=registry, indicators=indicator_traces, sub_indicators=subplot_traces,
    )
    appdata.save_data()

    # End the timer and print the execution time
    end_time = tm.perf_counter()
    print(f'Execution time: {end_time - start_time:.2f} seconds.')
