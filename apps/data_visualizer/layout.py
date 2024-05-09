from apps.data_visualizer.data import Data
from datetime import datetime
from dash import dcc

import dash_bootstrap_components as dbc
import numpy as np

# Load data.
DATA = Data.load_data()
# Extract unique dates from the OHLC data
DATES = np.unique(DATA.ohlc['data'].index.date)

def create_date_picker_single(id: str, default_date: datetime) -> dcc.DatePickerSingle:
    """
    Create a single date picker component.

    Parameters:
    id (str): The id of the date picker component.
    default_date (datetime): The default date selected in the date picker.

    Returns:
    dcc.DatePickerSingle: A Dash DatePickerSingle component.
    """
    return dcc.DatePickerSingle(
        id=id,
        min_date_allowed=DATES[0],
        max_date_allowed=DATES[-1],
        placeholder='Start Date',
        date=default_date,
        calendar_orientation='vertical',
        style={'margin-left': '0px'},
    )

def create_header(
        id: str,
        date_picker_start_id: str,
        default_start_date: datetime,
        date_picker_end_id: str,
        default_end_date: datetime,
) -> dbc.Container:
    """
    Create a header container with two date pickers.

    Parameters:
    id (str): The id of the header container.
    date_picker_start_id (str): The id of the start date picker.
    default_start_date (datetime): The default start date selected in the start date picker.
    date_picker_end_id (str): The id of the end date picker.
    default_end_date (datetime): The default end date selected in the end date picker.

    Returns:
    dbc.Container: A Dash Bootstrap Container component.
    """
    return dbc.Container(
        id=id,
        style={'width': '100%', 'height': '50px', 'margin-left': '0px', 'padding-left': '0px'},
        children=[
            create_date_picker_single(date_picker_start_id, default_date=default_start_date),
            create_date_picker_single(date_picker_end_id, default_date=default_end_date),
        ]
    )

# Check if there are any trades in the data
if DATA.trade_reg is not None:
    # Create the figure and table for the backtest results
    RESULT_FIGURE, RESULT_TABLE = DATA.create_result_figure()

    # Create a header for the backtest results. This includes two date pickers and a reload button.
    header_backtest_result = create_header(
        id='backtest-result-header',
        date_picker_start_id='backtest-result-date-picker-start',
        default_start_date=DATES[0],
        date_picker_end_id='backtest-result-date-picker-end',
        default_end_date=DATES[-1],
    )
    # Add a button to the header to reload the data
    header_backtest_result.children.append(
        dbc.Button(
            'Reload Data',
            color='primary',
            id='reload-backtest-result-button',
            className='me-1',
            style={'margin-left': '15px'},
        ))

    # Create a tab for the backtest results. This includes a container with the header and a row with the table and graph.
    backtest_result_tab = dbc.Tab(
        id='backtest-result-tab',
        label='Backtest Result',
        children=[

            dbc.Container(
                id='backtest-result-tab-content',
                style={'margin-left': '0px', 'margin-right': '0px', 'width': '1776px'},
                className='dbc',
                children=[

                    # Add the header to the container
                    header_backtest_result,

                    # Create a row with the table and graph
                    dbc.Row(
                        style={'height': '900px', 'width': '1800px'},
                        children=[

                            # Create a column for the table
                            dbc.Col(
                                id='backtest-result-table',
                                style={'height': '900px', 'width': '400px'},
                                width=3,
                                children=[
                                    RESULT_TABLE
                                ]),

                            # Create a column for the graph
                            dbc.Col(
                                style={'padding-left': '50px', 'height': '900px', 'width': '1400px'},
                                width=8,
                                className='dbc',
                                children=[

                                    # Add the graph to the column
                                    dcc.Graph(
                                        id='backtest-result-graph',
                                        config={'scrollZoom': True},
                                        style={'height': '900px', 'width': '100%'},
                                        figure=RESULT_FIGURE,
                                    ),

                                ]),
                        ])
                ])
        ]
    )

# Create a header for the market data. This includes two date pickers and a reload button.
header_market_data = create_header(
    id='market-data-header',
    date_picker_start_id='market-data-date-picker-start',
    default_start_date=DATES[0],
    date_picker_end_id='market-data-date-picker-end',
    default_end_date=DATES[1],
)
# Add a button to the header to reload the data
header_market_data.children.append(
    dbc.Button(
        'Reload Data',
        color='primary',
        id='reload-market-data-button',
        className='me-1',
        style={'margin-left': '15px'},
))

# Create a tab for the market data. This includes a container with the header, a checklist for the daily separator, and a graph.
market_data_tab = dbc.Tab(
    id='market-data-tab',
    label='Market Data',
    children=[
        dbc.Container(
            fluid=True,
            className='dbc',
            children=[
                # Add the header to the container
                header_market_data,

                # Create a checklist for the daily separator
                dbc.Checklist(
                    id='daily-separator-toggle',
                    value=[],
                    switch=True,
                    options=[
                        {'label': 'Show Daily Separator', 'value': 'show-daily-separator'},
                    ],
                ),

                # Add a graph to the container
                dcc.Graph(
                    id='market-data',
                    config={'scrollZoom': True},
                    style={'width': '100%', 'height': '850px'},
                ),
            ])
    ],
)

# Create the main container for the layout. This includes a set of tabs.
layout = dbc.Container(
    id='main-container',
    fluid=True,
    style={'width': '1800px', 'max-width': '1800px', 'height': '1000px', 'padding-left': '12px'},
    children=[
        dbc.Tabs(
            id='tabs',
            # The active tab is 'backtest-result-tab' if there are any trades in the data, otherwise it is 'market-data-tab'.
            children=[backtest_result_tab, market_data_tab] if DATA.trade_reg is not None else [market_data_tab],
        ),
    ],
)
