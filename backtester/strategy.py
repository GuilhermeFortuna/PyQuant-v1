from backtester.trades import Trade, TradeOrder
from abc import ABC, abstractmethod
from backtester.data import OHLC
from typing import Union, Tuple

import pandas_ta as pta
import optuna as op


class TradingStrategy(ABC):
    """
    Abstract base class for a trading strategy.

    This class should be subclassed when creating a new trading strategy. It provides the structure for the strategy
    by defining the methods that need to be implemented.

    Methods
    -------
    __init__(self)
        Initializes the TradingStrategy instance. This method is currently a placeholder and does not perform any actions.
    compute_indicators(self, ohlc: OHLC) -> None
        Abstract method for computing indicators. This method should be overridden in subclasses to compute the indicators
        required for the specific trading strategy.
    entry_strategy(self, i: int, ohlc: OHLC) -> Union[TradeOrder, None]
        Abstract method for defining the entry strategy. This method should be overridden in subclasses to define the
        conditions under which a trade should be entered.
    exit_strategy(self, i: int, ohlc: OHLC, trade: Trade) -> Union[TradeOrder, None]
        Abstract method for defining the exit strategy. This method should be overridden in subclasses to define the
        conditions under which a trade should be exited.
    """

    def __init__(self):
        """
        Initializes the TradingStrategy instance.

        This method is currently a placeholder and does not perform any actions.
        """
        pass

    @abstractmethod
    def compute_indicators(self, ohlc: OHLC) -> None:
        """
        Abstract method for computing indicators.

        This method should be overridden in subclasses to compute the indicators required for the specific trading strategy.

        Parameters:
        ohlc (OHLC): The OHLC data for the trading period.
        """
        pass

    @abstractmethod
    def entry_strategy(self, i: int, ohlc: OHLC) -> Union[TradeOrder, None]:
        """
        Abstract method for defining the entry strategy.

        This method should be overridden in subclasses to define the conditions under which a trade should be entered.

        Parameters:
        i (int): The index of the current trading period.
        ohlc (OHLC): The OHLC data for the trading period.

        Returns:
        TradeOrder or None: A TradeOrder object if a trade should be entered, otherwise None.
        """
        pass

    @abstractmethod
    def exit_strategy(self, i: int, ohlc: OHLC, trade: Trade) -> Union[TradeOrder, None]:
        """
        Abstract method for defining the exit strategy.

        This method should be overridden in subclasses to define the conditions under which a trade should be exited.

        Parameters:
        i (int): The index of the current trading period.
        ohlc (OHLC): The OHLC data for the trading period.
        trade (Trade): The current trade.

        Returns:
        TradeOrder or None: A TradeOrder object if the trade should be exited, otherwise None.
        """
        pass

class MACrossover(TradingStrategy):
    """
    A class that represents a Moving Average Crossover trading strategy.

    This class is a concrete implementation of the TradingStrategy abstract base class. It uses two moving averages,
    a short-term and a long-term, to generate trading signals. A buy signal is generated when the short-term moving
    average crosses above the long-term moving average, and a sell signal is generated when the short-term moving
    average crosses below the long-term moving average.

    Attributes
    ----------
    MA_FUNCS : dict
        A dictionary mapping the names of moving average functions to their corresponding functions in the pandas_ta library.
    short_ma_func : function
        The function to compute the short-term moving average.
    long_ma_func : function
        The function to compute the long-term moving average.
    short_ma_period : int
        The period for the short-term moving average.
    long_ma_period : int
        The period for the long-term moving average.
    delta_quantile : float
        The quantile used to compute the delta threshold.
    delta_window : int
        The window size used to compute the delta threshold.

    Methods
    -------
    __init__(self, short_ma_func: str = 'ema', long_ma_func: str = 'sma', short_ma_period: int = 9, long_ma_period: int = 12, delta_quantile: float = 0.3, delta_window: int = 50)
        Initializes the MACrossover with the given parameters.
    compute_indicators(self, ohlc: OHLC) -> None
        Computes the indicators required for the trading strategy.
    entry_strategy(self, i: int, ohlc: OHLC) -> Union[TradeOrder, None]
        Defines the entry strategy for the trading strategy.
    exit_strategy(self, i: int, ohlc: OHLC, trade: Trade) -> Union[TradeOrder, None]
        Defines the exit strategy for the trading strategy.
    optimize_parameters(self, trial: op.Trial, ohlc: OHLC, point_value: float, cost_per_trade: float) -> Tuple[float, float]
        Optimizes the parameters of the trading strategy using Optuna.
    """

    MA_FUNCS = {
        'sma': pta.sma,
        'ema': pta.ema,
        'jma': pta.jma,
    }

    def __init__(
            self,
            short_ma_func: str = 'ema',
            long_ma_func: str = 'sma',
            short_ma_period: int = 9,
            long_ma_period: int = 12,
            delta_quantile: float = 0.3,
            delta_window: int = 50,
    ):
        """
        Initializes the MACrossover with the given parameters.

        Parameters:
        short_ma_func (str): The name of the function to compute the short-term moving average. Default is 'ema'.
        long_ma_func (str): The name of the function to compute the long-term moving average. Default is 'sma'.
        short_ma_period (int): The period for the short-term moving average. Default is 9.
        long_ma_period (int): The period for the long-term moving average. Default is 12.
        delta_quantile (float): The quantile used to compute the delta threshold. Default is 0.3.
        delta_window (int): The window size used to compute the delta threshold. Default is 50.
        """

        self.short_ma_func = self.MA_FUNCS[short_ma_func]
        self.long_ma_func = self.MA_FUNCS[long_ma_func]
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.delta_quantile = delta_quantile
        self.delta_window = delta_window

    def compute_indicators(self, ohlc: OHLC) -> None:
        """
        Computes the indicators required for the trading strategy.

        This method computes the short-term and long-term moving averages and their difference (delta). It also computes
        the delta threshold, which is the delta_quantile quantile of the absolute delta over a rolling window of size delta_window.

        Parameters:
        ohlc (OHLC): The OHLC data for the trading period.
        """

        candle_data = ohlc.data
        candle_data['short_ma'] = self.short_ma_func(candle_data['close'], self.short_ma_period)
        candle_data['long_ma'] = self.long_ma_func(candle_data['close'], self.long_ma_period)
        candle_data['delta'] = candle_data['short_ma'] - candle_data['long_ma']
        candle_data['delta_threshold'] = \
            candle_data['delta'].abs().rolling(self.delta_window).quantile(self.delta_quantile)

    def entry_strategy(self, i: int, ohlc: OHLC) -> Union[TradeOrder, None]:
        """
        Defines the entry strategy for the trading strategy.

        This method generates a buy signal when the delta crosses above the delta threshold, and a sell signal when the
        delta crosses below the negative delta threshold.

        Parameters:
        i (int): The index of the current trading period.
        ohlc (OHLC): The OHLC data for the trading period.

        Returns:
        TradeOrder or None: A TradeOrder object if a trade should be entered, otherwise None.
        """

        order = None

        delta = ohlc.delta[i]
        prior_delta = ohlc.delta[i - 1]

        delta_thresh = ohlc.delta_threshold[i]
        prior_delta_thresh = ohlc.delta_threshold[i - 1]

        if prior_delta < prior_delta_thresh and delta > delta_thresh:
            order = TradeOrder(
                type='buy',
                price=ohlc.close[i],
                datetime=ohlc.datetime_index[i],
                comment='',
                amount=1,
            )

        elif prior_delta > -prior_delta_thresh and delta < -delta_thresh:
            order = TradeOrder(
                type='sell',
                price=ohlc.close[i],
                datetime=ohlc.datetime_index[i],
                comment='',
                amount=1,
            )

        return order

    def exit_strategy(self, i: int, ohlc: OHLC, trade: Trade) -> Union[TradeOrder, None]:
        """
        Defines the exit strategy for the trading strategy.

        This method generates a sell signal for a buy trade when the delta crosses below the negative delta threshold,
        and a buy signal for a sell trade when the delta crosses above the delta threshold.

        Parameters:
        i (int): The index of the current trading period.
        ohlc (OHLC): The OHLC data for the trading period.
        trade (Trade): The current trade.

        Returns:
        TradeOrder or None: A TradeOrder object if the trade should be exited, otherwise None.
        """

        order = None

        if trade.type == 'buy':
            if ohlc.delta[i] < -ohlc.delta_threshold[i]:
                order = TradeOrder(
                    type='invert',
                    price=ohlc.close[i],
                    datetime=ohlc.datetime_index[i],
                    comment='',
                    amount=1,
                )

        elif trade.type == 'sell':
            if ohlc.delta[i] > ohlc.delta_threshold[i]:
                order = TradeOrder(
                    type='invert',
                    price=ohlc.close[i],
                    datetime=ohlc.datetime_index[i],
                    comment='',
                    amount=1,
                )

        return order

    def optimize_parameters(
            self,
            trial: op.Trial,
            ohlc: OHLC,
            point_value: float,
            cost_per_trade: float,
    ) -> Tuple[float, float]:
        """
        Optimizes the parameters of the trading strategy using Optuna.

        This method uses Optuna to find the optimal parameters for the trading strategy. It runs a backtest for each
        set of parameters and returns the net balance, drawdown, profit factor, and accuracy of the best set of parameters.

        Parameters:
        trial (op.Trial): The trial object from Optuna.
        ohlc (OHLC): The OHLC data for the trading period.
        point_value (float): The point value for the backtest.
        cost_per_trade (float): The cost per trade for the backtest.

        Returns:
        Tuple[float, float]: The net balance and drawdown of the best set of parameters.
        """

        from backtester.engine import Engine

        trial_penalty = -100_000, 100_000

        short_ma_func = trial.suggest_categorical('short_ma_func', list(MACrossover.MA_FUNCS.keys()))
        long_ma_func = trial.suggest_categorical('long_ma_func', list(MACrossover.MA_FUNCS.keys()))
        short_ma_period = trial.suggest_int('short_ma_period', 1, 50)
        long_ma_period = trial.suggest_int('long_ma_period', 1, 50)
        delta_quantile = trial.suggest_float('delta_quantile', 0.2, 0.9, step=0.1)

        self.short_ma_func = self.MA_FUNCS[short_ma_func]
        self.long_ma_func = self.MA_FUNCS[long_ma_func]
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.delta_quantile = delta_quantile
        self.delta_window = 50

        engine = Engine(point_value=point_value, cost_per_trade=cost_per_trade)
        result_data = engine.run_backtest(ohlc=ohlc, strategy=self, bypass_first_exit_check=True)
        processed_data, registry = result_data['data'], result_data['registry']
        result = registry.get_result(verbose=True, as_dataframe=False)

        # Get objective values and display them along with parameter values used.
        if result is not None:
            balance, drawdown = result['net_balance (BRL)'], result['drawdown_percentage (%)']
            profit_factor, accuracy = result['profit_factor'], result['accuracy (%)']
            print(f'Parameters: {self.__dict__}\n\nBalance: {balance:.2f}\nDrawdown: {drawdown:.2f}%\nProfit Factor: '
                  f'{profit_factor:.2f}\nAccuracy: {accuracy:.2f}\n')
            print('#' * 75)

            return balance, drawdown, profit_factor, accuracy

        else:
            return trial_penalty