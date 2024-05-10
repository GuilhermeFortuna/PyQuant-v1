from backtester.dashboards.data_visualizer.layout import layout
from backtester.dashboards.data_visualizer.data import Data
from dash_bootstrap_templates import load_figure_template
from dash import Dash, callback, Input, Output, State
from datetime import datetime

import dash_bootstrap_components as dbc

# Load the darkly template from dash bootstrap templates
load_figure_template('darkly')

# Create a Dash application with the darkly theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Set the layout of the application
app.layout = layout

# Define a callback function to update the 'market-data' figure
# This function is triggered by the 'reload-market-data-button'
# It uses the start and end dates and the daily separator value as state
@callback(
    Output('market-data', 'figure'),
    Input('reload-market-data-button', 'n_clicks'),
    State('market-data-date-picker-start', 'date'),
    State('market-data-date-picker-end', 'date'),
    State('daily-separator-toggle', 'value'),
)
def slice_figure(n_clicks, start_date, end_date, daily_separator):
    """
    Update the 'market-data' figure based on the selected start and end dates and the daily separator value.

    Parameters:
    n_clicks (int): The number of times the 'reload-market-data-button' has been clicked.
    start_date (str): The selected start date.
    end_date (str): The selected end date.
    daily_separator (list): The value of the 'daily-separator-toggle'.

    Returns:
    plotly.graph_objs._figure.Figure: The updated figure.
    """

    # Parse the start and end dates
    start_date = start_date.split('T')[0]
    end_date = end_date.split('T')[0]

    # Determine whether to show the daily separator
    daily_separator = True if isinstance(daily_separator, list) and len(daily_separator) > 0  and \
                              daily_separator[0] == 'show-daily-separator' else False

    # Convert the start and end dates to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Load the data and create the figure
    data = Data.load_data()
    return data.create_data_figure(start_date=start_date, end_date=end_date, date_separator=daily_separator)

# Define a callback function to update the 'backtest-result-graph' figure and the 'backtest-result-table'
# This function is triggered by the 'reload-backtest-result-button'
# It uses the start and end dates as state
@callback(
    Output('backtest-result-graph', 'figure'),
    Output('backtest-result-table', 'children'),
    Input('reload-backtest-result-button', 'n_clicks'),
    State('backtest-result-date-picker-start', 'date'),
    State('backtest-result-date-picker-end', 'date'),
    config_prevent_initial_callbacks=True,
)
def slice_backtest_result(n_clicks, start_date, end_date):
    """
    Update the 'backtest-result-graph' figure and the 'backtest-result-table' based on the selected start and end dates.

    Parameters:
    n_clicks (int): The number of times the 'reload-backtest-result-button' has been clicked.
    start_date (str): The selected start date.
    end_date (str): The selected end date.

    Returns:
    Tuple[plotly.graph_objs._figure.Figure, dash_html_components.Div]: The updated figure and table.
    """

    # Parse and convert the start and end dates to datetime objects
    start_date = start_date.split('T')[0]
    start_date = datetime.strptime(start_date, '%Y-%m-%d')

    end_date = end_date.split('T')[0]
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Load the data and create the figure and table
    data = Data.load_data()
    return data.create_result_figure(start_date=start_date, end_date=end_date)

if __name__ == '__main__':
    # Run the Dash application in debug mode on port 8050.
    # Debug mode provides more detailed error messages and enables hot reloading,
    # which means the server will automatically refresh whenever a change is made to the source code.
    app.run(debug=True, port=8050)
