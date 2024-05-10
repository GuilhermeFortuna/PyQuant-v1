from backtester.trades import TradeRegistry, TradeOrder
from backtester.strategy import TradingStrategy
from backtester.data import OHLC
from typing import Dict, Union
from tqdm import tqdm


class Engine:
    """
    A class that represents the engine for running backtests.

    This class is responsible for running backtests using a given trading strategy on OHLC data. It uses a TradeRegistry
    to keep track of trades during the backtest.

    Attributes
    ----------
    registry : TradeRegistry
        The TradeRegistry used to keep track of trades during the backtest.

    Methods
    -------
    __init__(self, point_value: float, cost_per_trade: float)
        Initializes the Engine with the given point value and cost per trade.
    _run_backtest(self, ohlc: OHLC, strategy: TradingStrategy, display_progress: bool = True, bypass_first_exit_check: bool = False) -> Dict[OHLC, TradeRegistry]
        Runs a backtest using the given strategy on the OHLC data. This method is intended to be used internally.
    run_backtest(self, ohlc: OHLC, strategy: TradingStrategy, display_progress: bool = True, permit_swingtrade: bool = True, bypass_first_exit_check: bool = False) -> Dict[OHLC, TradeRegistry]
        Runs a backtest using the given strategy on the OHLC data. This method is the public interface for running backtests.
    """

    def __init__(self, point_value: float, cost_per_trade: float):
        """
        Initializes the Engine with the given point value and cost per trade.

        Parameters:
        point_value (float): The point value of the trades.
        cost_per_trade (float): The cost per trade.

        Raises:
        ValueError: If point_value or cost_per_trade is not an int or float.
        """

        if not isinstance(point_value, (int, float)):
            raise ValueError('The point_value parameter must be an integer or a float.')

        if not isinstance(cost_per_trade, (int, float)):
            raise ValueError('The cost_per_trade parameter must be an integer or a float.')

        self.registry = TradeRegistry(point_value=point_value, cost_per_trade=cost_per_trade)

    def _run_backtest(
            self,
            ohlc: OHLC,
            strategy: TradingStrategy,
            display_progress: bool = True,
            bypass_first_exit_check: bool = False,
    ) -> Dict[OHLC, TradeRegistry]:
        """
        Runs a backtest using the given strategy on the OHLC data.

        This method is intended to be used internally. It runs a backtest by iterating over the OHLC data and applying
        the entry and exit strategies of the given TradingStrategy. It uses a TradeRegistry to keep track of trades during
        the backtest.

        Parameters:
        ohlc (OHLC): The OHLC data for the backtest.
        strategy (TradingStrategy): The trading strategy to use for the backtest.
        display_progress (bool): Whether to display a progress bar during the backtest. Default is True.
        bypass_first_exit_check (bool): Whether to bypass the first exit check after entering a trade. Default is False.

        Returns:
        Dict[OHLC, TradeRegistry]: A dictionary containing the OHLC data and the TradeRegistry after the backtest.
        """

        registry, trade = self.registry, self.registry.trade

        if display_progress:
            print('\n\n', '#' * 75, '\n')
            pbar = tqdm(total=len(ohlc.index), desc='Running backtest...', colour='green', leave=None)

        # Run backtest.
        for i in ohlc.index:

            is_last_candle = i == ohlc.index[-1]

            if trade.status is None and not is_last_candle:
                order: Union[TradeOrder, None] = strategy.entry_strategy(i, ohlc)

                if order is not None:
                    registry.process_order(order)

                    if bypass_first_exit_check:
                        continue

            if trade.status == 'open':
                order: Union[TradeOrder, None] = strategy.exit_strategy(i, ohlc, trade)

                if order is not None:
                    registry.process_order(order)

            if is_last_candle and trade.status == 'open':
                order = TradeOrder(
                    type='close',
                    price=ohlc.close[i],
                    datetime=ohlc.datetime_index[i],
                    comment='Last candle',
                    amount=trade._amount,
                )
                registry.process_order(order)

            if display_progress:
                pbar.update(1)

        return {'data': ohlc, 'registry': registry}

    def run_backtest(
            self,
            ohlc: OHLC,
            strategy: TradingStrategy,
            display_progress: bool = True,
            permit_swingtrade: bool = True,
            bypass_first_exit_check: bool = False,
    ) -> Dict[OHLC, TradeRegistry]:
        """
        Runs a backtest using the given strategy on the OHLC data.

        This method is the public interface for running backtests. It first validates the input parameters, then computes
        the indicators required by the trading strategy. If swing trading is permitted, it runs the backtest on the entire
        OHLC data. If not, it runs the backtest on each day of the OHLC data separately.

        Parameters:
        ohlc (OHLC): The OHLC data for the backtest.
        strategy (TradingStrategy): The trading strategy to use for the backtest.
        display_progress (bool): Whether to display a progress bar during the backtest. Default is True.
        permit_swingtrade (bool): Whether to permit swing trading. Default is True.
        bypass_first_exit_check (bool): Whether to bypass the first exit check after entering a trade. Default is False.

        Returns:
        Dict[OHLC, TradeRegistry]: A dictionary containing the OHLC data and the TradeRegistry after the backtest.

        Raises:
        ValueError: If any of the input parameters are of the wrong type.
        """

        if not isinstance(ohlc, OHLC):
            raise ValueError('The ohlc parameter must be an instance of the OHLC class.')

        if not isinstance(strategy, TradingStrategy):
            raise ValueError('The strategy parameter must be an instance of the TradingStrategy class.')

        if not isinstance(display_progress, bool):
            raise ValueError('The display_progress parameter must be a boolean value.')

        if not isinstance(permit_swingtrade, bool):
            raise ValueError('The permit_swingtrade parameter must be a boolean value.')

        if not isinstance(bypass_first_exit_check, bool):
            raise ValueError('The bypass_first_exit_check parameter must be a boolean value.')

        trades = self.registry
        trades.trade._reset()

        # Compute indicators from trading strategy.
        strategy.compute_indicators(ohlc)
        ohlc._set_values_as_attributes()

        if not permit_swingtrade:
            if display_progress:
                pbar = tqdm(total=len(ohlc.index), desc='Running backtest...', colour='green', leave=None)

            processed_data, registries = [], []
            for day in OHLC.group_by_date(ohlc.data, symbol=ohlc.symbol, timeframe=ohlc.timeframe):
                result = self._run_backtest(
                    ohlc=day,
                    strategy=strategy,
                    display_progress=display_progress,
                    bypass_first_exit_check=bypass_first_exit_check,
                )
                processed_data.append(result['data'])
                registries.append(result['registry'])

                if display_progress:
                    pbar.update(1)

        else:
            return self._run_backtest(
                ohlc=ohlc,
                strategy=strategy,
                display_progress=display_progress,
                bypass_first_exit_check=bypass_first_exit_check,
            )
