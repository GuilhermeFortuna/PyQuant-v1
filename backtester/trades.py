'''
    Module responsible for handling trades.
'''

from typing import Optional, Union, Tuple
from datetime import datetime, date
from collections import namedtuple

import pandas as pd
import numpy as np

import warnings
import math


TradeOrder = namedtuple('TradeOrder', ['type', 'price', 'datetime', 'comment', 'amount'])

class Trade:
    """
    A class to represent a trade.

    Attributes
    ----------
    point_value : float
        The point value of the trade.
    cost_per_trade : float
        The cost per trade.
    _start : datetime, optional
        The start time of the trade.
    _end : datetime, optional
        The end time of the trade.
    _amount : int, optional
        The amount of the trade.
    _type : str, optional
        The type of the trade (buy or sell).
    _buyprice : float, optional
        The buy price of the trade.
    _sellprice : float, optional
        The sell price of the trade.
    _delta : float, optional
        The delta of the trade.
    _result : float, optional
        The result of the trade.
    _cost : float, optional
        The cost of the trade.
    _profit : float, optional
        The profit of the trade.
    _duration : str, optional
        The duration of the trade.
    _entry_comment : str, optional
        The entry comment of the trade.
    _exit_comment : str, optional
        The exit comment of the trade.
    _status : str, optional
        The status of the trade.
    _order_history : list, optional
        The order history of the trade.
    _position_size : dict, optional
        The position size of the trade.

    Methods
    -------
    __init__(self, point_value: float, cost_per_trade: float)
        Initializes the Trade with the given point value and cost per trade.
    """

    def __init__(self, point_value: float, cost_per_trade: float):
        """
        Initializes the Trade with the given point value and cost per trade.

        Parameters:
        point_value (float): The point value of the trade.
        cost_per_trade (float): The cost per trade.

        Raises:
        TypeError: If point_value or cost_per_trade is not an int or float.
        """

        if not isinstance(point_value, (int, float)):
            raise TypeError('Expected point_value to be an int or float.')

        if not isinstance(cost_per_trade, (int, float)):
            raise TypeError('Expected cost_per_trade to be an int or float.')

        self.point_value = point_value
        self.cost_per_trade = cost_per_trade

        self._start = None
        self._end = None
        self._amount = None
        self._type = None
        self._buyprice = None
        self._sellprice = None
        self._delta = None
        self._result = None
        self._cost = None
        self._profit = None
        self._duration = None
        self._entry_comment = None
        self._exit_comment = None

        self._status = None
        self._order_history = []
        self._position_size = {
            'type': None,  # Type of trade. str, either 'long' or 'short'.
            'size': None,  # Size of position. int, any positive integer.
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the Trade instance.

        Returns:
        str: A string representation of the Trade instance.
        """
        return f'Trade({self._start}, {self._end}, {self._amount}, {self._type}, {self._buyprice}, {self._sellprice},' \
               f' {self._delta}, {self._result}, {self._cost}, {self._profit}, {self._duration},' \
               f' {self._entry_comment}, {self._exit_comment})'

    @property
    def status(self) -> str:
        """
        Returns the status of the trade.

        Returns:
        str: The status of the trade.
        """
        return self._status

    @property
    def start(self) -> Optional[datetime]:
        """
        Returns the start datetime of the trade.

        Returns:
        datetime: The start datetime of the trade.
        """
        return self._start

    @property
    def end(self) -> Optional[datetime]:
        """
        Returns the end datetime of the trade.

        Returns:
        datetime: The end datetime of the trade.
        """
        return self._end

    @property
    def amount(self) -> Optional[int]:
        """
        Returns the amount of the trade.

        Returns:
        int: The amount of the trade.
        """
        return self._amount

    @property
    def type(self) -> Optional[str]:
        """
        Returns the type of the trade.

        Returns:
        str: The type of the trade.
        """
        return self._type

    @property
    def buyprice(self) -> Optional[float]:
        """
        Returns the buy price of the trade.

        Returns:
        float: The buy price of the trade.
        """
        return self._buyprice

    @property
    def sellprice(self) -> Optional[float]:
        """
        Returns the sell price of the trade.

        Returns:
        float: The sell price of the trade.
        """
        return self._sellprice

    @property
    def delta(self) -> Optional[float]:
        """
        Returns the delta of the trade.

        Returns:
        float: The delta of the trade.
        """
        return self._delta

    @property
    def result(self) -> Optional[float]:
        """
        Returns the result of the trade.

        Returns:
        float: The result of the trade.
        """
        return self._result

    @property
    def cost(self) -> Optional[float]:
        """
        Returns the cost of the trade.

        Returns:
        float: The cost of the trade.
        """
        return self._cost

    @property
    def profit(self) -> Optional[float]:
        """
        Returns the profit of the trade.

        Returns:
        float: The profit of the trade.
        """
        return self._profit

    @property
    def duration(self) -> Optional[str]:
        """
        Returns the duration of the trade.

        Returns:
        str: The duration of the trade.
        """
        return self._duration

    @property
    def entry_comment(self) -> Optional[str]:
        """
        Returns the entry comment of the trade.

        Returns:
        str: The entry comment of the trade.
        """
        return self._entry_comment

    @property
    def exit_comment(self) -> Optional[str]:
        """
        Returns the exit comment of the trade.

        Returns:
        str: The exit comment of the trade.
        """
        return self._exit_comment

    def _update_position(self, order: TradeOrder) -> None:
        """
        Updates the position size based on the given order.

        Parameters:
        order (TradeOrder): The order to update the position size with.

        Raises:
        ValueError: If the order type is not valid.
        """

        # Check if order type is valid.
        if order.type not in ['buy', 'sell', 'close']:
            raise ValueError(f'Invalid order type: {order.type} for status: {self._status}')

        # Update position size.
        # Open position.
        if self._position_size['type'] is None:
            self._position_size['type'] = order.type
            self._position_size['size'] = self._amount = order.amount

            # Update status.
            self._status = 'open'

        # Close position.
        elif order.type == 'close':
            self._position_size['size'] = 0
            self._position_size['type'] = None

            # Update status. Compute trade result.
            self._status = 'closed'
            self._compute_result()

        # Increase position size.
        elif self._position_size['type'] == order.type:
            self._position_size['size'] += order.amount

            # Update max position size.
            if self._position_size['size'] > self._amount:
                self._amount = self._position_size['size']

        # Decrease position size.
        elif self._position_size['type'] != order.type:
            self._position_size['size'] -= order.amount

            # Close position.
            if self._position_size['size'] == 0:
                self._position_size['type'] = None

                # Update status. Compute trade result.
                self._status = 'closed'
                self._compute_result()

    def _open_trade(self, order: TradeOrder) -> None:
        """
        Opens a trade based on the given order.

        Parameters:
        order (TradeOrder): The order to open the trade with.

        Raises:
        ValueError: If the order type is not valid.
        """

        # Check if order type is valid.
        if order.type not in ['buy', 'sell']:
            raise ValueError(f'Invalid order type: {order.type} for status: {self._status}')

        # Register trade information.
        self._start = order.datetime
        self._type = order.type
        self._entry_comment = order.comment

        if order.type == 'buy':
            self._buyprice = order.price
        elif order.type == 'sell':
            self._sellprice = order.price

        # Update position size.
        self._update_position(order)

    def _close_trade(self, order: TradeOrder) -> None:
        """
        Closes a trade based on the given order.

        Parameters:
        order (TradeOrder): The order to close the trade with.

        Raises:
        ValueError: If the order type is not valid.
        """

        # Check if order type is valid.
        if order.type != 'close':
            raise ValueError(f'Invalid order type: {order.type} for status: {self._status}')

        # Register trade information.
        self._end = order.datetime
        self._exit_comment = order.comment

        if self._type == 'buy':
            self._sellprice = order.price
        elif self._type == 'sell':
            self._buyprice = order.price

        # Update position size.
        self._update_position(order)

    def _invert_trade(self, order: TradeOrder) -> None:
        """
        Inverts a trade based on the given order.

        This method gets the current position type, closes the current position, and then opens a new position with the
        inverted type based on the previous position type. It then updates the position size.

        Parameters:
        order (TradeOrder): The order to invert the trade with.

        Note: This method is intended to be used internally and may not provide the expected results if used outside of its intended context.
        """

        # Get info on current position. Close position.
        current_position_type = self.type
        self._close_trade(TradeOrder(
            type='close', price=order.price, datetime=order.datetime, comment=order.comment, amount=self._amount,
        ))

        # Invert position based on previous position type.
        if current_position_type == 'buy':
            invert_order = TradeOrder(
                type='sell', price=order.price, datetime=order.datetime, comment=order.comment, amount=order.amount
            )
            self._open_trade(invert_order)

        elif current_position_type == 'sell':
            invert_order = TradeOrder(
                type='buy', price=order.price, datetime=order.datetime, comment=order.comment, amount=order.amount
            )
            self._open_trade(invert_order)

        # Update position size.
        self._update_position(invert_order)

    def _compute_result(self):
        """
        Computes the result of the trade.

        This method calculates the delta, result, cost, and profit of the trade.
        It also calculates the duration of the trade in days, hours, and minutes.
        """

        # Compute trade result.
        self._delta = self._sellprice - self._buyprice
        self._result = self._delta * self.point_value * self._amount
        self._cost = self.cost_per_trade
        self._profit = self._result - self._cost

        # Compute trade duration.
        duration = self._end - self._start
        self._duration = (f'{duration.days} days {duration.seconds // 3600} hours'
                          f' {duration.seconds % 3600 // 60} minutes')

    def process_order(self, order: TradeOrder) -> None:
        """
        Processes a trade order.

        This method validates the order, adds it to the order history, and then processes it based on the current status of the trade.

        Parameters:
        order (TradeOrder): The order to be processed.

        Raises:
        TypeError: If the order is not an instance of TradeOrder.
        NotImplementedError: If the order type is not supported.
        """

        # Validate argument.
        if not isinstance(order, TradeOrder):
            raise TypeError(f'Expected TradeOrder, got {type(order)}')

        # Add order to order history.
        self._order_history.append(order)

        # Open trade.
        if self._status is None:
            self._open_trade(order)

        # Manage open trade.
        elif self._status == 'open':

            # Close position.
            if order.type == 'close':
                self._close_trade(order)

            # Invert position.
            elif order.type == 'invert':
                self._invert_trade(order)

            # Increase position size.
            elif self._type == order.type:
                raise NotImplementedError('Increasing position size is not yet implemented.')

            # Decrease position size.
            elif self._type != order.type:
                raise NotImplementedError('Decreasing position size is not yet implemented.')

        elif self._status == 'closed':
            pass

    def _as_dataframe(self) -> pd.DataFrame:
        """
        Returns trade information as a pandas DataFrame.

        This method creates a DataFrame with the trade information and returns it.

        Returns:
        pd.DataFrame: A DataFrame with the trade information.
        """

        trade_info = pd.DataFrame({
            'start': [self._start],
            'end': [self._end],
            'amount': [self._amount],
            'type': [self._type],
            'buyprice': [self._buyprice],
            'sellprice': [self._sellprice],
            'delta': [self._delta],
            'result': [self._result],
            'cost': [self._cost],
            'profit': [self._profit],
            'duration': [self._duration],
            'entry_comment': [self._entry_comment],
            'exit_comment': [self._exit_comment],
        })

        return trade_info

    def _reset(self) -> None:
        """
        Resets the trade information.

        This method resets all the trade information attributes to their initial state.
        """

        self._start = None
        self._end = None
        self._amount = None
        self._type = None
        self._buyprice = None
        self._sellprice = None
        self._delta = None
        self._result = None
        self._cost = None
        self._profit = None
        self._duration = None
        self._entry_comment = None
        self._exit_comment = None

        self._status = None
        self._position_size = {
            'type': None,
            'size': None,
            'max_size': None,
        }

class TradeRegistry:

    def __init__(self, point_value: float, cost_per_trade: float):
        self.point_value = point_value
        self.cost_per_trade = cost_per_trade

        self.trade = Trade(point_value=point_value, cost_per_trade=cost_per_trade)
        self.trades = pd.DataFrame(
            columns=['start', 'end', 'amount', 'type', 'buyprice', 'sellprice', 'delta', 'result', 'cost',
                     'profit', 'balance', 'entry_comment', 'exit_comment'],
        )

    @property
    def net_balance(self) -> float:
        '''
        Returns net balance.
        '''
        return round(self.trades['balance'].iat[-1], 2)

    @property
    def num_positive_trades(self) -> int:
        '''
        Returns number of positive trades.
        '''
        return self.trades.loc[self.trades['result'] > 0, 'result'].count()

    @property
    def num_negative_trades(self) -> int:
        '''
        Returns number of negative trades.
        '''
        return self.trades.loc[self.trades['result'] < 0, 'result'].count()

    @property
    def positive_trade_sum(self) -> float:
        '''
        Returns sum of positive trades.
        '''
        return round(self.trades.loc[self.trades['result'] > 0, 'result'].sum(), 2)

    @property
    def negative_trade_sum(self) -> float:
        '''
        Returns sum of negative trades.
        '''
        return round(self.trades.loc[self.trades['result'] < 0, 'result'].sum(), 2)

    @property
    def profit_factor(self) -> float:
        '''
        Returns profit factor.
        '''

        profit_factor = self.positive_trade_sum / abs(self.negative_trade_sum) if self.negative_trade_sum != 0 else \
            math.inf if self.positive_trade_sum > 0 else 0
        return round(profit_factor, 2)

    @property
    def accuracy(self) -> float:
        '''
        Returns trade accuracy.
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('Accuracy cannot be calculated as there are no trades.')
            return

        accuracy = (self.num_positive_trades / len(self.trades)) * 100
        return round(accuracy, 2)

    @property
    def mean_profit(self) -> float:
        '''
        Returns mean profit.
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('Mean profit cannot be calculated as there are no trades.')
            return

        mean_profit = self.positive_trade_sum / self.num_positive_trades if self.num_positive_trades != 0 else 0
        return round(mean_profit, 2)

    @property
    def mean_loss(self) -> float:
        '''
        Returns mean loss.
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('Mean loss cannot be calculated as there are no trades.')
            return

        mean_loss = self.negative_trade_sum / self.num_negative_trades if self.num_negative_trades != 0 else 0
        return round(mean_loss, 2)

    @property
    def mean_profit_loss_ratio(self) -> float:
        '''
        Returns mean profit to loss ratio.
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('Mean profit loss ratio cannot be calculated as there are no trades.')
            return

        ratio = self.mean_profit / abs(self.mean_loss) if self.mean_loss != 0 else math.inf if self.mean_profit != 0 \
            else 0
        return round(ratio, 2)

    @property
    def result_standard_deviation(self) -> float:
        '''
        Returns result standard deviation.
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('Result standard deviation cannot be calculated as there are no trades.')
            return

        return round(self.trades['result'].std(), 2)

    @classmethod
    def join_trades(cls, registries: list):
        '''
        Takes a list of TradeRegistries and joins them into one.

        :param registries: list[TradeRegistry]. List of TradeRegistry instances.
        :return: TradeRegistry.
        '''

        # Check if instances is not empty.
        if len(registries) == 0:
            raise ValueError('Instances must contain at least one instance of TradeRegistry.')

        # Create instance of class.
        reg = registries[0]
        registry = cls(point_value=reg.point_value, cost_per_trade=reg.cost_per_trade)

        # Join trades.
        trades_list = [x.trades for x in registries]
        registry.trades = pd.concat([*trades_list], axis='index', ignore_index=True)
        registry.trades.sort_values(by='end', ignore_index=True, inplace=True)
        registry.trades['balance'] = registry.trades['profit'].cumsum()

        return registry

    def _register_trade(self) -> None:
        self.trades = pd.concat([self.trades, self.trade._as_dataframe()], ignore_index=True, sort=False)
        self.trade._reset()
        self.trades['balance'] = self.trades['profit'].cumsum()

    def process_order(self, order: TradeOrder) -> None:

        if order.type == 'invert':
            current_position_type = self.trade.type
            self.trade._close_trade(TradeOrder(
                type='close', price=order.price, datetime=order.datetime, comment=order.comment, amount=order.amount,
            ))
            self._register_trade()

            if current_position_type == 'buy':
                invert_order = TradeOrder(
                    type='sell', price=order.price, datetime=order.datetime, comment=order.comment, amount=order.amount
                )
                self.trade._open_trade(invert_order)

            elif current_position_type == 'sell':
                invert_order = TradeOrder(
                    type='buy', price=order.price, datetime=order.datetime, comment=order.comment, amount=order.amount
                )
                self.trade._open_trade(invert_order)

        else:
            self.trade.process_order(order)
            if self.trade._status == 'closed':
                self._register_trade()

    def trades_today(self, date: date) -> int:
        '''
        Returns number of trades today.
        '''
        return len(self.trades.loc[self.trades['start'].dt.date == date])

    def _compute_maximum_drawdown(self, percentage_method: str = 'relative') -> Union[Tuple[float, float], None]:
        '''
        Compute maximum drawdown.

        :param percentage_method: str. The method to use for computing the drawdown percentage. Options are 'relative'
        and 'final'.
        :return: Tuple[float, float]. The maximum drawdown as a float and as a percentage value.
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('No registered trades. Unable to get maximum drawdown.')
            return None

        # Compute maximum drawdown.
        dd = self.trades[['balance']].copy()
        dd['max_balance'] = dd['balance'].cummax()
        dd['max_balance'] = dd['max_balance'].mask(dd['max_balance'] < 0, 0)
        dd['drawdown'] = dd['max_balance'] - dd['balance']
        max_drawdown = dd['drawdown'].max()
        if percentage_method == 'relative':
            drawdown_percentage = (max_drawdown / dd['max_balance'].iat[dd['drawdown'].idxmax()]) * 100

        elif percentage_method == 'final':
            drawdown_percentage = (max_drawdown / dd['balance'].iat[-1]) * 100

        else:
            raise ValueError('Invalid percentage method. Options are "relative" and "final".')

        return round(max_drawdown, 2), round(drawdown_percentage, 2)

    def compute_monthly_result(self, return_df: bool = False) -> Optional[pd.DataFrame]:
        '''
        Get result by month.

        :param return_df: bool. Whether to return a dataframe.
        :return: Optional[pd.DataFrame].
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('No registered trades. Unable to get monthly result.')
            return None

        # Copy trades dataframe and add month column to use for groupby operation.
        trades = self.trades.copy()
        trades['month'] = trades['end'].map(lambda x: x.date().replace(day=1))
        monthly_group = trades.groupby(by='month')

        # Create monthly result dataframe.
        monthly_index = np.unique(trades['month'])
        result = pd.DataFrame(index=monthly_index, columns=['num_trades', 'result', 'cost', 'profit', 'balance'])

        # Compute monthly result.
        for date, group in monthly_group:
            result.at[date, 'num_trades'] = len(group)
            result.at[date, 'result'] = round(group['result'].sum(), 2)
            result.at[date, 'cost'] = round(group['cost'].sum(), 2)
            result.at[date, 'profit'] = round(group['profit'].sum(), 2)

        result.at[date, 'balance'] = result['profit'].cumsum().round(2)
        self.monthly_result = result

        if return_df:
            return result

    def monthly_result_stats(self) -> dict:
        '''
        Get monthly result stats.

        :return: dict.
        '''

        # Check if monthly result has been computed. If not, attempt to compute.
        if self.monthly_result is None:
            if not self.trades.empty:
                self.compute_monthly_result()

            else:
                raise RuntimeError('Attempting to compute monthly result when trades dataframe is empty.')

        # Create stats dictionary.
        stats = {}

        # Get monthly result.
        monthly_result = self.monthly_result
        positive_months = monthly_result.loc[monthly_result['profit'] > 0].copy()
        negative_months = monthly_result.loc[monthly_result['profit'] < 0].copy()

        # Compute stats.
        stats['positive_months'] = len(positive_months)
        stats['negative_months'] = len(negative_months)
        stats['positive_ratio'] = (stats['positive_months'] / len(monthly_result)) * 100 if \
            stats['negative_months'] != 0 else 100
        stats['mean_positive_months'] = positive_months['profit'].mean()
        stats['mean_negative_months'] = negative_months['profit'].mean()
        stats['best_month'] = positive_months['profit'].max()
        stats['worst_month'] = negative_months['profit'].min()
        stats['std_deviation'] = monthly_result['profit'].std()

        return stats

    def get_result(
            self,
            verbose: bool = False,
            plot_results: bool = False,
            as_dataframe: bool = True,
            include_monthly_stats: bool = False,
            drawdown_percentage_method: str = 'relative',
    ) -> Union[pd.DataFrame, dict, None]:


        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('No registered trades. Unable to get result.')
            return

        max_dd, dd_pct = self._compute_maximum_drawdown(percentage_method=drawdown_percentage_method)
        self.compute_monthly_result()
        monthly_result = self.monthly_result
        result = {
            'net_balance (BRL)': self.net_balance,
            'gross_balance (BRL)': self.net_balance,
            'profit_factor': self.profit_factor,
            'accuracy (%)': self.accuracy,
            'mean_profit (BRL)': self.mean_profit,
            'mean_loss (BRL)': self.mean_loss,
            'mean_ratio': self.mean_profit_loss_ratio,
            'standard_deviation': self.result_standard_deviation,
            'total_trades': len(self.trades),
            'positive_trades': self.num_positive_trades,
            'negative_trades': self.num_negative_trades,
            'maximum_drawdown (BRL)': max_dd,
            'drawdown_percentage (%)': dd_pct,
            'start_date': self.trades['start'].iat[0],
            'end_date': self.trades['end'].iat[-1],
            'duration': self.trades['end'].iat[-1] - self.trades['start'].iat[0],
            'average_monthly_result (BRL)': round(monthly_result['profit'].mean(), 2),
        }

        monthly_stats = self.monthly_result_stats() if include_monthly_stats else {}

        # Print result if verbose.
        if verbose:
            print('\n\n--- Result ---\n')
            for metric, value in result.items():
                print(f'{metric}:'.ljust(30), f'{value}'.rjust(25))

            if include_monthly_stats:
                print('\n')
                print('-' * 75)
                print('\n--- Monthly Stats ---\n')
                for stat, value in monthly_stats.items():
                    print(f'{stat}:'.ljust(30), f'{round(value, 2)}'.rjust(25))

            print('\n\n')

        # Plot results if 'plot_results' is True.
        if plot_results:
            raise NotImplementedError('Plotting not yet implemented.')

        result.update(monthly_stats)
        if as_dataframe:
            return pd.DataFrame.from_dict(result, orient='index', columns=['result'])

        return result
