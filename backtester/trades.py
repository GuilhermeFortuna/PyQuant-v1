'''
    Module responsible for handling trades.
'''

from typing import Optional, Union, Tuple
from collections import OrderedDict
from datetime import datetime, date
from dataclasses import dataclass

import pandas as pd
import numpy as np

import warnings
import math

@dataclass
class TradeOrder:
    '''
    Class responsible for passing trade information.
    '''

    type: str
    price: float
    datetime: datetime
    comment: str = ''
    amount: Optional[int] = None
    slippage: Optional[float] = None

class TradeRegistry:

    def __init__(self, point_value: float, cost_per_trade: float, tax_rate: Optional[float] = None):
        self.point_value = point_value
        self.cost_per_trade = cost_per_trade
        self.tax_rate = tax_rate

        self.trades = pd.DataFrame(
            columns=['start', 'end', 'amount', 'type', 'buyprice', 'sellprice', 'delta', 'result', 'cost',
                     'profit', 'balance', 'entry_comment', 'exit_comment'],
        )

    @property
    def net_balance(self) -> float:
        '''
        Returns net balance.
        '''

        # Check if all trades have been processed. If not, process trades.
        if self.trades['balance'].isna().any():
            self._process_trades()

        return round(self.trades['balance'].iat[-1], 2)

    @property
    def num_positive_trades(self) -> int:
        '''
        Returns number of positive trades.
        '''

        # Check if all trades have been processed. If not, process trades.
        if self.trades['result'].isna().any():
            self._process_trades()

        return self.trades.loc[self.trades['result'] > 0, 'result'].count()

    @property
    def num_negative_trades(self) -> int:
        '''
        Returns number of negative trades.
        '''

        # Check if all trades have been processed. If not, process trades.
        if self.trades['result'].isna().any():
            self._process_trades()

        return self.trades.loc[self.trades['result'] < 0, 'result'].count()

    @property
    def positive_trade_sum(self) -> float:
        '''
        Returns sum of positive trades.
        '''

        # Check if all trades have been processed. If not, process trades.
        if self.trades['result'].isna().any():
            self._process_trades()

        return round(self.trades.loc[self.trades['result'] > 0, 'result'].sum(), 2)

    @property
    def negative_trade_sum(self) -> float:
        '''
        Returns sum of negative trades.
        '''

        # Check if all trades have been processed. If not, process trades.
        if self.trades['result'].isna().any():
            self._process_trades()

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

    @property
    def _last_trade_index(self) -> int:
        '''
        Returns index of last trade.
        '''
        return len(self.trades.index) - 1 if not self.trades.empty else 0

    @property
    def _new_trade_index(self) -> int:
        '''
        Returns index to register new trade.
        '''
        return len(self.trades.index) if not self.trades.empty else 0

    @property
    def open_trade_info(self) -> Union[dict, None]:
        '''
        Get information on current open trade. Returns None if no trade is open.

        :return: Union[dict, None].
        '''

        last_trade_idx = self._last_trade_index
        if not self.trades.empty and isinstance(self.trades.at[last_trade_idx, 'start'], datetime) and \
                not isinstance(self.trades.at[last_trade_idx, 'end'], datetime):

            trade_info = {}
            trade_info['type'] = self.trades.at[last_trade_idx, 'type']
            trade_info['price'] = self.trades.at[last_trade_idx, 'buyprice'] if trade_info['type'] == 'buy' else \
                self.trades.at[last_trade_idx, 'sellprice']
            trade_info['datetime'] = self.trades.at[last_trade_idx, 'start']
            trade_info['comment'] = self.trades.at[last_trade_idx, 'entry_comment']

            return trade_info

    @classmethod
    def join_trades(cls, registries: list):
        '''
        Takes a list of TradeRegistries and joins them into one.

        :param registries: list[TradeRegistry].
        :param combined_registry: bool.
        :return: TradeRegistry.
        '''

        # Check if instances is not empty.
        if len(registries) == 0:
            raise ValueError('Instances must contain at least one instance of TradeRegistry.')

        # Create instance of class.
        reg = registries[0]
        registry = cls(
            point_value=reg.point_value, cost_per_trade=reg.cost_per_trade, daytrade_tax_rate=reg.daytrade_tax_rate,
            swingtrade_tax_rate=reg.swingtrade_tax_rate,
        )

        # Join trades.
        trades_list = [x.trades for x in registries]
        registry.trades = pd.concat([*trades_list], axis='index', ignore_index=True)
        registry.trades.sort_values(by='end', ignore_index=True, inplace=True)
        registry._process_trades(force_process=True)

        return registry

    def _buy(self, order: TradeOrder) -> None:
        '''
        Register buy trade.

        :param order: TradeOrder.
        :return: None.
        '''

        # Register buy in trades dataframe.
        index = self._new_trade_index
        self.trades.at[index, 'type'] = 'buy'
        self.trades.at[index, 'buyprice'] = order.price if order.slippage is None else order.price + order.slippage
        self.trades.at[index, 'start'] = order.datetime
        self.trades.at[index, 'entry_comment'] = order.comment
        self.trades.at[index, 'amount'] = order.amount

    def _sell(self, order: TradeOrder) -> None:
        '''
        Register sell trade.

        :param order: TradeOrder.
        :return: None.
        '''

        # Register sell in trades dataframe.
        index = self._new_trade_index
        self.trades.at[index, 'type'] = 'sell'
        self.trades.at[index, 'sellprice'] = order.price if order.slippage is None else order.price - order.slippage
        self.trades.at[index, 'start'] = order.datetime
        self.trades.at[index, 'entry_comment'] = order.comment
        self.trades.at[index, 'amount'] = order.amount

    def _close_position(self, order: TradeOrder) -> None:
        '''
        Close the last open position.

        :param order: TradeOrder.
        :return: None.
        '''

        # Get info on open trade.
        open_trade = self.open_trade_info
        idx = self._last_trade_index

        # Close an existing buy position.
        if open_trade['type'] == 'buy':
            self.trades.at[idx, 'sellprice'] = order.price if order.slippage is None else order.price + order.slippage
            self.trades.at[idx, 'end'] = order.datetime

        # Close an existing sell position.
        if open_trade['type'] == 'sell':
            self.trades.at[idx, 'buyprice'] = order.price if order.slippage is None else order.price - order.slippage
            self.trades.at[idx, 'end'] = order.datetime

        # Register exit comment.
        self.trades.at[idx, 'exit_comment'] = order.comment

    def trades_today(self, date: date) -> int:
        '''
        Returns number of trades today.
        '''
        return len(self.trades.loc[pd.DatetimeIndex(self.trades['start']).date == date])

    def register_order(self, order: TradeOrder) -> None:
        '''
        Register order in trades dataframe.

        :param order: TradeOrder.
        :return: None.
        '''

        # Add order to order history.
        order_num = len(self.order_history)
        self.order_history[order_num] = order

        # Open buy position.
        if order.type == 'buy':
            if self.open_trade_info is None:
                self._buy(order)
            else:
                raise RuntimeError('Attempting to register a buy trade when a position is already open.')

        # Open sell position.
        elif order.type == 'sell':
            if self.open_trade_info is None:
                self._sell(order)
            else:
                raise RuntimeError('Attempting to register a sell trade when a position is already open.')

        # Close position.
        elif order.type == 'close':
            if self.open_trade_info is None:
                raise RuntimeError('Attempting to register a close trade when there is no open position.')
            else:
                self._close_position(order)

        # Invert position.
        elif order.type == 'invert':
            if self.open_trade_info is None:
                raise RuntimeError('Attempting to register an invert trade when there is no open position.')
            else:
                trade_info = self.open_trade_info

                if trade_info['type'] == 'buy':
                    self._close_position(order)
                    self._sell(order)

                elif trade_info['type'] == 'sell':
                    self._close_position(order)
                    self._buy(order)

        # Invalid order type.
        else:
            raise ValueError(f'Invalid order type: {order.type}')

    def _compute_tax(self, df: pd.DataFrame) -> float:
        '''
        Intended to be used with pd.DataFrameGroupBy.apply(). Compute tax for each month group.

        :param df: pd.DataFrame.
        :return: float.
        '''

        trade_start = pd.DatetimeIndex(df['start'])
        trade_end = pd.DatetimeIndex(df['end'])
        daytrade_indices = df.index[np.where(trade_start.day == trade_end.day)]
        swingtrade_indices = df.index.difference(daytrade_indices)

        daytrade = df.loc[daytrade_indices, 'profit'].sum()
        swingtrade = df.loc[swingtrade_indices, 'profit'].sum()
        d_tax = daytrade * 0.2 if daytrade > 0 else 0
        s_tax = swingtrade * 0.15 if swingtrade > 0 else 0
        total_tax = d_tax + s_tax

        return total_tax

    def _process_trades(self, force_process: bool = False) -> None:
        '''
        Process trades dataframe. Compute results.

        :return: None.
        '''

        # Check if already processed.
        if not self.trades.isna().any().any() and not force_process:
            return

        # Process trade data.
        self.trades['delta'] = (self.trades['sellprice'] - self.trades['buyprice']).astype(float).round(decimals=2)
        self.trades['result'] = (self.trades['delta'] * self.point_value * self.trades['amount']).astype(float).round(
            decimals=2)
        self.trades['cost'] = self.cost_per_trade * self.trades['amount']
        self.trades['profit'] = (self.trades['result'] - self.trades['cost']).astype(float).round(decimals=2)
        self.trades['balance'] = (self.trades['profit'].cumsum()).astype(float).round(decimals=2)

        self.trades['entry_comment'] = self.trades['entry_comment'].astype(str)
        self.trades['exit_comment'] = self.trades['exit_comment'].astype(str)
        self.trades['entry_comment'] = self.trades['entry_comment'].fillna('')
        self.trades['exit_comment'] = self.trades['exit_comment'].fillna('')

        # Compute tax.
        traded_months = self.trades['end'].map(lambda x: x.date().replace(day=1))
        monthly_trade_groups = self.trades.groupby(traded_months)
        self.tax_per_month = monthly_trade_groups.apply(self._compute_tax)
        self.total_tax = round(self.tax_per_month.sum(), 2)

    def _compute_maximum_drawdown(self, percentage_method: str = 'relative') -> Tuple[float, float]:
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
        result = pd.DataFrame(index=monthly_index, columns=['num_trades', 'result', 'cost', 'tax', 'profit',
                                                            'balance'])

        # Compute monthly result.
        for date, group in monthly_group:
            monthly_tax = round(self.tax_per_month.at[date])
            result.at[date, 'num_trades'] = len(group)
            result.at[date, 'result'] = round(group['result'].sum(), 2)
            result.at[date, 'cost'] = round(group['cost'].sum(), 2)
            result.at[date, 'tax'] = monthly_tax
            result.at[date, 'profit'] = round(group['profit'].sum() - monthly_tax, 2)
            result.at[date, 'balance'] = round(group['balance'].iloc[-1], 2)

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
            force_process_trades: bool = False,
            export_to_excel: bool = False,
    ) -> Union[pd.DataFrame, dict, None]:
        '''
        Get compiled result.

        :param drawdown_method: str. The method to use for computing the maximum drawdown. Options are 'peak_balance'
        and 'final_balance'.
        :param verbose: bool. Whether to print the result.
        :param plot_results: bool. Whether to plot the results.
        :param as_dataframe: bool. Whether to return the result as a pandas DataFrame. Otherwise, returns a dict.
        :param include_monthly_stats: bool. Whether to include monthly stats in the result.
        :param drawdown_percentage_method: str. The method to use for computing the drawdown percentage. Options are
        'relative' and 'final'.
        :param export_to_excel: bool. Whether to export the result to an Excel file. The file will be exported as
        'backtest_result.xlsx' to the current working directory.
        :return: Union[pd.DataFrame, dict].
        '''

        # Check if trades is not empty.
        if self.trades.empty:
            warnings.warn('No registered trades. Unable to get result.')
            return

        self._process_trades(force_process=force_process_trades)
        max_dd, dd_pct = self._compute_maximum_drawdown(percentage_method=drawdown_percentage_method)
        self.compute_monthly_result()
        monthly_result = self.monthly_result
        result = {
            'net_balance (BRL)': round(self.net_balance - self.total_tax, 2),
            'gross_balance (BRL)': self.net_balance,
            'total_tax (BRL)': self.total_tax,
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
            print('\n\n--- Results ---\n')
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
