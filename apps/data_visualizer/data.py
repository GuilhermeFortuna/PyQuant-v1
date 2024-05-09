from backtester.trades import TradeRegistry
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
from datetime import datetime

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

import pickle

class Data:
    """
    The Data class is responsible for loading, saving, and processing data for the application.
    It handles OHLC data, volume at price data, trade registry data, and indicator data.
    """

    DATA_PATH = r'C:\Users\Gui\PycharmProjects\PyQuant_v1\apps\data_visualizer\data\data.pkl'

    def __init__(
            self,
            ohlc: pd.DataFrame,
            vap_table: Optional[pd.DataFrame] = None,
            trade_registry: Optional[TradeRegistry] = None,
            indicators: Optional[List[go.Scatter]] = None,
            sub_indicators: Optional[List[go.Scatter]] = None,
    ):
        """
        Initialize the Data object.

        Parameters:
        ohlc (pd.DataFrame): The OHLC data.
        vap_table (Optional[pd.DataFrame]): The volume at price data.
        trade_registry (Optional[TradeRegistry]): The trade registry data.
        indicators (Optional[List[go.Scatter]]): The indicator data.
        sub_indicators (Optional[List[go.Scatter]]): The sub-indicator data.
        """

        if not isinstance(ohlc, pd.DataFrame):
            raise TypeError('ohlc must be a pandas DataFrame.')

        if vap_table is not None and not isinstance(vap_table, pd.DataFrame):
            raise TypeError('vap_table must be a pandas DataFrame.')

        if indicators is not None and not isinstance(indicators, list):
            raise TypeError('indicators must be a list.')

        if sub_indicators is not None and not isinstance(sub_indicators, list):
            raise TypeError('sub_indicators must be a list.')

        self.trade_reg = trade_registry

        # Process ohlc data.
        self.ohlc = {
            'trace': self.create_candlestick_trace(ohlc),
            'data': ohlc,
        }

        # Process volume at price.
        self.vap = {}
        if vap_table is not None:
            self.vap['trace'] = self.create_heatmap_trace(vap_table)
            self.vap['data'] = vap_table

        # Process indicators.
        self.indicators = {}
        if indicators is not None:
            for trace in indicators:
                x, y, name = trace.x, trace.y, trace.name
                data = pd.Series(y, index=x)

                self.indicators[name] = {
                    'trace': trace,
                    'data': data,
                }

        # Process sub indicators.
        self.sub_indicators = {}
        if sub_indicators is not None:
            for trace in sub_indicators:
                x, y, name = trace.x, trace.y, trace.name
                data = pd.Series(y, index=x)

                self.sub_indicators[name] = {
                    'trace': trace,
                    'data': data,
                }

    @classmethod
    def load_data(cls) -> None:
        """
        Load data from file.

        Returns:
        Data: The loaded data.
        """

        with open(Data.DATA_PATH, 'rb') as f:
            file_data = pickle.load(f)

        data = cls.__new__(cls)
        for key, value in file_data.items():
            setattr(data, key, value)

        return data

    def _load_data(self) -> None:
        """
        Load data from file.

        Returns:
        None.
        """

        with open(Data.DATA_PATH, 'rb') as f:
            file_data = pickle.load(f)

        for key, value in file_data.items():
            setattr(self, key, value)

    def save_data(self) -> None:
        """
        Save data to file.

        Returns:
        None.
        """

        data = {
            'ohlc': self.ohlc,
            'vap': self.vap,
            'trade_reg': self.trade_reg,
            'indicators': self.indicators,
            'sub_indicators': self.sub_indicators,
        }

        with open(Data.DATA_PATH, 'wb') as f:
            pickle.dump(data, f)

    def create_candlestick_trace(self, df: pd.DataFrame) -> go.Candlestick:
        """
        Create candlestick trace.

        Parameters:
        df (pd.DataFrame): The OHLC data.

        Returns:
        go.Candlestick: The candlestick trace.
        """

        return go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candles',
        )

    def create_heatmap_trace(self, df: pd.DataFrame) -> go.Heatmapgl:
        """
        Create heatmap trace.

        Parameters:
        df (pd.DataFrame): The volume at price data.

        Returns:
        go.Heatmapgl: The heatmap trace.
        """

        return go.Heatmap(
            z=df.T.values,
            x=df.index,
            y=df.columns,
            colorscale='thermal',
            showscale=False,
            opacity=0.5,
        )

    def create_data_figure(self, start_date: datetime, end_date: datetime, date_separator: bool = True) -> go.Figure:
        """
        Create a data figure based on the selected start and end dates and the daily separator value.

        Parameters:
        start_date (datetime): The selected start date.
        end_date (datetime): The selected end date.
        date_separator (bool): Whether to show the daily separator.

        Returns:
        go.Figure: The data figure.
        """

        from apps.data_visualizer.config import market_data_layout, xaxis

        # Load the data
        self._load_data()

        # Slice the OHLC data based on the selected start and end dates
        sliced_ohlc = self.ohlc['data'].loc[start_date:end_date]

        # Create a figure with the market data layout and update the y-axis range based on the sliced OHLC data
        fig = go.Figure(layout=market_data_layout)
        fig.update_yaxes(range=[sliced_ohlc['low'].min(), sliced_ohlc['high'].max()])

        # Create a candlestick trace with the sliced OHLC data and add it to the figure
        ohlc_trace = self.create_candlestick_trace(sliced_ohlc)
        fig.add_trace(ohlc_trace)

        # If there is volume at price data, create a heatmap trace and add it to the figure
        if isinstance(self.vap, dict) and len(self.vap) > 0:
            vap = self.vap['data'].loc[start_date:end_date].copy()
            vap.mask(vap == 0).dropna(axis=0, how='all', inplace=True)
            heatmap_trace = self.create_heatmap_trace(vap)
            fig.add_trace(heatmap_trace)

        # If there are indicators, create traces for each one and add them to the figure
        if len(self.indicators) > 0:
            for indicator in self.indicators.values():
                sliced_data =  indicator['data'].loc[start_date:end_date]
                trace = indicator['trace']
                trace.x = sliced_data.index
                trace.y = sliced_data.values

                fig.add_trace(trace)

        # If there are trades, create traces for each one and add them to the figure
        if isinstance(self.trade_reg, TradeRegistry) and isinstance(self.trade_reg.trades, pd.DataFrame) and \
                len(self.trade_reg.trades) > 0:
            trades = self.trade_reg.trades
            sliced_trades = trades.loc[(trades['start'] >= start_date) & (trades['end'] <= end_date)].copy()
            trades_traces = self.get_trades_trace(
                trades=sliced_trades, ohlc_index=sliced_ohlc.index, date_from=start_date, date_to=end_date,
            )
            fig.add_traces(trades_traces)

        # If there are sub-indicators, create traces for each one and add them to the figure
        if len(self.sub_indicators) > 0:

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2], figure=fig,
            )

            for indicator in self.sub_indicators.values():
                sliced_data =  indicator['data'].loc[start_date:end_date]
                trace = indicator['trace']
                trace.x = sliced_data.index
                trace.y = sliced_data.values

                fig.add_trace(trace, row=2, col=1)

            fig.update_xaxes(matches='x', row=2, col=1)
            fig.update_yaxes(showgrid=False, row=2, col=1)
            fig.update_layout(xaxis2=xaxis)

        # If the daily separator is enabled, add a shape for each date in the sliced OHLC data
        if date_separator:
            candle_groups = sliced_ohlc.groupby(sliced_ohlc.index.date)
            dates = candle_groups.nth(0).index
            for dt in dates:
                fig.add_shape(
                    dict(
                        x0=dt, x1=dt, y0=0, y1=1, xref='x', yref='paper',
                        line_width=1, line_color='darkgrey', type='line', line=dict(dash='dash'), opacity=0.5
                    )
                )

        return fig

    def get_trades_trace(
            self,
            trades: pd.DataFrame,
            ohlc_index: pd.DatetimeIndex,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None,
    ) -> list:
        """
        Get the traces of trades for a given period.

        Parameters:
        trades (pd.DataFrame): The trades data.
        ohlc_index (pd.DatetimeIndex): The index of the OHLC data.
        date_from (Optional[datetime]): The start date of the period. If None, it uses the start date of the trades data.
        date_to (Optional[datetime]): The end date of the period. If None, it uses the end date of the trades data.

        Returns:
        list: A list of Scatter traces for the trades.
        """

        from apps.data_visualizer.config import trade_marker_symbols, trade_marker_colors

        def create_trace(df: pd.DataFrame) -> go.Scatter:
            """
            Create a Scatter trace for the given data.

            Parameters:
            df (pd.DataFrame): The data for the trace.

            Returns:
            go.Scatter: The Scatter trace.
            """
            return go.Scatter(
                x=df.index,
                y=df.values,
                mode='markers',
                marker=dict(
                    size=20,
                    color=trade_marker_colors[df.name],
                    symbol=trade_marker_symbols[df.name],
                    line=dict(width=1, color='white')
                ),
                name=df.name,
            )

        # Snap trades to ohlc index.
        start_map = np.searchsorted(ohlc_index, trades['start'], 'right') - 1
        end_map = np.searchsorted(ohlc_index, trades['end'], 'right') - 1
        trades['start'] = ohlc_index[start_map]
        trades['end'] = ohlc_index[end_map]

        # Initialize the list of traces.
        traces = []
        if date_from is not None or date_to is not None:
            trades = trades[(trades['start'] >= date_from) & (trades['end'] <= date_to)].copy()

            # If trades is empty, return empty list.
            if trades.empty:
                return traces

        # Get trade entry info in dataframe.
        entry_buy_indices = pd.DatetimeIndex(trades.loc[trades['type'] == 'buy', 'start'])
        entry_sell_indices = pd.DatetimeIndex(trades.loc[trades['type'] == 'sell', 'start'])
        entry_buy = pd.Series(
            trades.loc[trades['type'] == 'buy', 'buyprice'].values, index=entry_buy_indices, name='entry_buy'
        )
        entry_sell = pd.Series(
            trades.loc[trades['type'] == 'sell', 'sellprice'].values, index=entry_sell_indices, name='entry_sell'
        )

        # Get trade exit info in dataframe.
        exit_buy_indices = trades.loc[trades['type'] == 'buy', 'end']
        exit_sell_indices = trades.loc[trades['type'] == 'sell', 'end']
        exit_buy = pd.Series(
            trades.loc[trades['type'] == 'buy', 'sellprice'].values, index=exit_buy_indices, name='exit_buy'
        )
        exit_sell = pd.Series(
            trades.loc[trades['type'] == 'sell', 'buyprice'].values, index=exit_sell_indices, name='exit_sell'
        )

        # Get trade traces.
        traces.append(create_trace(entry_buy))
        traces.append(create_trace(entry_sell))
        traces.append(create_trace(exit_buy))
        traces.append(create_trace(exit_sell))
        traces.extend(self.create_trade_interpolation_traces(idx=ohlc_index, trades=trades))

        return traces

    def create_trade_interpolation_traces(self, idx: pd.DatetimeIndex, trades: pd.DataFrame) -> list:
        """
        Create trade interpolation traces for a given index and trades data.

        Parameters:
        idx (pd.DatetimeIndex): The index of the OHLC data.
        trades (pd.DataFrame): The trades data.

        Returns:
        list: A list of Scatter traces for the trades.
        """

        # Helper function to map values
        def map_values(x, y) -> np.array:
            """
            Map values from x to y using searchsorted method.

            Parameters:
            x (array-like): The input array.
            y (array-like): The array to map to.

            Returns:
            np.array: The mapped values.
            """
            value_map = np.searchsorted(y, x, 'right') - 1
            return np.array([y[i] for i in value_map])

        # Copy trades data and map start and end to ohlc index
        t = trades.copy()
        t['start'] = map_values(t['start'], idx)
        t['end'] = map_values(t['end'], idx)

        # Initialize dataframe for trade interpolation
        trade_interp = pd.DataFrame(index=idx, columns=['pos', 'neg', 'result'], dtype='float')
        for i in t.index:
            start = t.at[i, 'start']
            end = t.at[i, 'end']
            entry_price = t.at[i, 'buyprice'] if t.at[i, 'type'] == 'buy' else t.at[i, 'sellprice']
            exit_price = t.at[i, 'buyprice'] if t.at[i, 'type'] == 'sell' else t.at[i, 'sellprice']

            # Fill trade_interp dataframe with trade results and interpolate prices
            trade_interp.loc[start: end, 'result'] = t.at[i, 'result']
            if t.at[i, 'result'] > 0:
                trade_interp.at[start, 'pos'] = entry_price
                trade_interp.at[end, 'pos'] = exit_price
                trade_interp.loc[start:end, 'pos'] = trade_interp.loc[start: end, 'pos'].interpolate()

            elif t.at[i, 'result'] <= 0:
                trade_interp.at[start, 'neg'] = entry_price
                trade_interp.at[end, 'neg'] = exit_price
                trade_interp.loc[start: end, 'neg'] = trade_interp.loc[start: end, 'neg'].interpolate()

        # Define colors for positive and negative trades
        trade_interp_colors = {
            'pos': 'green',
            'neg': 'red',
        }

        # Create Scatter traces for positive and negative trades
        trade_interp_traces = []
        for col in trade_interp[['pos', 'neg']]:
            trade_interp_traces.append(go.Scatter(
                x=idx,
                y=trade_interp[col],
                mode='lines',
                line=dict(color=trade_interp_colors[col], dash='dash'),
                name=f'{col}_trades',
                hovertext=trade_interp['result'],
            ))

        return trade_interp_traces

    def create_result_figure(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> Tuple[dbc.Table, go.Figure]:
        """
        Create a result figure and a table based on the start and end dates.

        Parameters:
        start_date (Optional[datetime]): The start date of the period. If None, it uses the start date of the trades data.
        end_date (Optional[datetime]): The end date of the period. If None, it uses the end date of the trades data.

        Returns:
        Tuple[dbc.Table, go.Figure]: The result table and figure.
        """

        # Import necessary modules and classes
        from apps.data_visualizer.config import result_layout
        from backtester.trades import TradeRegistry

        # Create a figure with two subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.4],
            row_titles=['Strategy Balance', 'Trade Results']
        )

        # If there is no trade registry, return the figure and an empty table
        if self.trade_reg is None:
            return fig, dbc.Table()

        # Get a copy of trades dataframe
        trades = self.trade_reg.trades.copy()

        # Filter trades by start and end datetime
        if start_date is None:
            start_date = trades['start'].iat[0]

        if end_date is None:
            end_date = trades['end'].iat[-1]

        # Slice trades and create trade registry with sliced trades
        trades = trades[(trades['start'] >= start_date) & (trades['end'] <= end_date.replace(hour=23))].copy()

        # If trades is empty, return the figure and an empty table
        if not isinstance(trades, pd.DataFrame) or trades.empty:
            return fig, dbc.Table()

        # Create a new trade registry with the sliced trades
        reg = TradeRegistry(point_value=self.trade_reg.point_value, cost_per_trade=self.trade_reg.cost_per_trade)
        reg.trades = trades

        # Compute expanding moving average of result
        result_ma = trades['result'].expanding().mean()

        # Get the result of the trade registry as a dataframe
        result_df = reg.get_result(as_dataframe=True, drawdown_percentage_method='final')
        result_df.reset_index(inplace=True)
        result_df.rename(columns={'index': 'Metric', 'result': 'Result'}, inplace=True)

        # Create a table from the result dataframe
        result_table = dbc.Table.from_dataframe(result_df, striped=True, bordered=True, hover=True)

        # Create a datetime index from the end dates of the trades
        index = pd.DatetimeIndex(trades['end'])

        # Add traces to the figure
        fig.add_trace(go.Scatter(
            x=index,
            y=trades['balance'],
            mode='lines',
            line=dict(
                color='yellow',
                width=2
            ),
            name='Net Balance'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=index,
            y=trades['result'].cumsum(),
            mode='lines',
            line=dict(
                color='white',
                width=1,
                dash='dash',
            ),
            name='Gross Balance'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=index,
            y=trades['result'],
            mode='markers',
            marker=dict(
                color=trades['result'],
                autocolorscale=False,
                colorscale='PiYG', size=5,
            ),
            name='Trade Results'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=index,
            y=result_ma,
            mode='lines',
            line=dict(
                color='yellow',
                width=2
            ),
            name='Result MA'
        ), row=2, col=1)

        # Update the layout of the figure
        fig.update_layout(
            **result_layout, xaxis2=dict(showgrid=False), yaxis2=dict(showgrid=False), title_text='Backtest Results',
            title_x=0.5, font=dict(color='white', size=14),
        )

        return fig, result_table
