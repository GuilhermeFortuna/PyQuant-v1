# Layout elements.
trade_marker_symbols = {
    'entry_buy': 'triangle-up',
    'entry_sell': 'triangle-down',
    'exit_buy': 'triangle-down',
    'exit_sell': 'triangle-up',
}

trade_marker_colors = {
    'entry_buy': 'green',
    'entry_sell': 'red',
    'exit_buy': 'yellow',
    'exit_sell': 'yellow',
}

xaxis = dict(
    type='category',
    # rangemode='tozero',
    nticks=9,
    showgrid=False,
    tickmode='auto',
    fixedrange=False,
    rangeslider={'visible': False},
    zeroline=True,
)
yaxis = dict(
    # nticks=7,
    showgrid=False,
    fixedrange=False,
    tickmode='auto',
    scaleanchor='x',
)

# Layouts.
market_data_layout = {
    'xaxis': xaxis,
    'yaxis': yaxis,
    #'height': 1000,
    'plot_bgcolor': 'black',
    'paper_bgcolor': 'black',
    'autosize': True,
    'hovermode': 'x unified',
    'hoverlabel': {
        'bgcolor': 'white',
        'font': {'color': 'gray'},
        'align': 'right',
    },
}
result_layout = {
    'xaxis': {
        'showgrid': False,
    },
    'yaxis': {
        'showgrid': False,
    },
    'plot_bgcolor': 'black',
    'paper_bgcolor': 'black',
    'autosize': True,
    'hoverlabel': {
        'bgcolor': 'white',
        'font': {'color': 'gray'},
        'align': 'right'
    },
    #'title': 'Strategy Gross Balance',
    #'title_x': 0.5,
}