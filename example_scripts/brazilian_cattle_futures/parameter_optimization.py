from backtester.strategy import MACrossover
from optuna_dashboard import run_server
from backtester.backtester.data import OHLC
from backtest import DATA_PARAMS
from datetime import datetime

import optuna as op
import functools

# This script is used to optimize the parameters of the MACrossover strategy.
# It uses the Optuna library for the optimization process and the Optuna Dashboard to visualize the results.

if __name__ == '__main__':

    # Set the date range for the data to be used in the optimization process.
    DATA_PARAMS['date_from'] = datetime(2019, 1, 1)
    DATA_PARAMS['date_to'] = datetime(2022, 1, 1)

    # Create an OHLC (Open, High, Low, Close) data object.
    # The symbol and timeframe are specified in the DATA_PARAMS dictionary.
    data = OHLC(symbol=DATA_PARAMS['symbol'], timeframe=DATA_PARAMS['timeframe'])

    # Load the OHLC data from a CSV file.
    # The file path and date range are specified in the DATA_PARAMS dictionary.
    data.load_data_from_csv(
        filepath=DATA_PARAMS['data_path'], date_from=DATA_PARAMS['date_from'], date_to=DATA_PARAMS['date_to'],
    )

    # Create an instance of the MACrossover strategy.
    strategy = MACrossover()

    # Define a function to optimize the parameters of the strategy.
    # The function is a partial application of the optimize_parameters method of the strategy,
    # with the OHLC data, point value, and cost per trade specified.
    optimize_params = functools.partial(
        strategy.optimize_parameters,
        ohlc=data,
        point_value=DATA_PARAMS['point_value'],
        cost_per_trade=DATA_PARAMS['cost_per_trade'],
    )

    # Create an Optuna study to perform the optimization.
    # The study uses an in-memory storage and has four objectives to maximize.
    storage = op.storages.InMemoryStorage()
    study = op.create_study(study_name='MACrossover', storage=storage, directions=['maximize', 'minimize'])

    # Run the optimization process with 100 trials.
    study.optimize(optimize_params, n_trials=100)

    # Start the Optuna Dashboard to visualize the results of the optimization.
    run_server(storage)
