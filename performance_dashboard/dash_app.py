import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash_table.Format import Format, Scheme
import plotly.graph_objs as go
import pandas as pd
import numpy as np



month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

periods_columns = ["WTD", "MTD", "YTD", "ITD", "1M", "3M", "6M", "1Y"]
# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def compute_cumulative_returns(returns_series: pd.Series) -> pd.Series:
    """Compute the cumulative returns over time."""
    return (1 + returns_series).cumprod() - 1

def compute_drawdown(cum_returns: pd.Series) -> pd.Series:
    """Compute drawdown from cumulative returns."""
    running_max = cum_returns.cummax()
    return (cum_returns / running_max) - 1

def annualized_sharpe_ratio(returns_series: pd.Series, freq=252) -> float:
    """Annualized Sharpe Ratio."""
    std = returns_series.std()
    if std == 0:
        return 0
    return (returns_series.mean() / std) * np.sqrt(freq)

def annualized_sortino_ratio(returns_series: pd.Series, freq=252) -> float:
    """Annualized Sortino Ratio."""
    negative_returns = returns_series[returns_series < 0]
    neg_std = negative_returns.std()
    if neg_std == 0:
        return 0
    return (returns_series.mean() / neg_std) * np.sqrt(freq)

def max_drawdown(returns_series: pd.Series) -> float:
    """Compute the maximum drawdown from a daily returns series."""
    cum = (1 + returns_series).cumprod()
    dd = compute_drawdown(cum)
    return dd.min()

def average_daily_return(returns_series: pd.Series) -> float:
    """Average daily return."""
    return returns_series.mean()

def hit_rate(returns_series: pd.Series) -> float:
    """Fraction of days with returns >= 0 (counting zero as positive)."""
    return (returns_series[returns_series!=0] >= 0).mean()

def period_performance(returns_series: pd.Series) -> float:
    """(1 + r_1)*(1 + r_2)*...*(1 + r_n) - 1 over the given period."""
    if len(returns_series) == 0:
        return 0
    return (1 + returns_series).prod() - 1

# -- For sub-period slicing in the performance table --
def get_period_slice(df: pd.DataFrame, period: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Helper to slice the (already date-filtered) DataFrame for WTD, MTD, YTD, etc.
    We'll interpret 'today' as 'end_date'.
    """
    if df.empty:
        return df
    
    valid_dates = df.index[(df.index <= end_date) & (df.index >= start_date)]
    if len(valid_dates) == 0:
        return df.iloc[0:0]  # empty
    actual_end_date = valid_dates[-1]

    # Start-of definitions
    start_of_week = actual_end_date - pd.to_timedelta(actual_end_date.weekday(), unit='D')
    start_of_month = pd.to_datetime(f"{actual_end_date.year}-{actual_end_date.month:02d}-01")
    start_of_year = pd.to_datetime(f"{actual_end_date.year}-01-01")
    global_start = valid_dates.min()

    if period == "WTD":
        start = max(global_start, start_of_week)
    elif period == "MTD":
        start = max(global_start, start_of_month)
    elif period == "YTD":
        start = max(global_start, start_of_year)
    elif period == "ITD":
        start = global_start
    elif period == "1M":  # last ~21 trading days
        idx = df.loc[:actual_end_date].index
        start_idx = max(0, len(idx) - 21)
        start = idx[start_idx] if len(idx) > 0 else actual_end_date
    elif period == "3M":  # last ~63 trading days
        idx = df.loc[:actual_end_date].index
        start_idx = max(0, len(idx) - 63)
        start = idx[start_idx] if len(idx) > 0 else actual_end_date
    elif period == "6M":  # last ~126 trading days
        idx = df.loc[:actual_end_date].index
        start_idx = max(0, len(idx) - 126)
        start = idx[start_idx] if len(idx) > 0 else actual_end_date
    elif period == "1Y":  # last ~126 trading days
        idx = df.loc[:actual_end_date].index
        start_idx = max(0, len(idx) - 252)
        start = idx[start_idx] if len(idx) > 0 else actual_end_date
    else:
        start = global_start

    return df.loc[(df.index >= start) & (df.index <= actual_end_date)]

def compute_performance_table(df: pd.DataFrame,start_date: pd.Timestamp,  end_date: pd.Timestamp):
    """
    For each strategy (column in df), compute:
    - WTD, MTD, YTD, Since Inception,
    - 1M, 3M, 6M, 1Y
    - Sharpe, Sortino, MaxDD, AvgDaily, HitRate,
    all restricted to the filtered data [start_date, end_date].
    """
    if df.empty:
        return []

    table_rows = []
    
    for strategy in df.columns:
        row_data = {"Strategy": strategy}
        for p in periods_columns:
            sliced = get_period_slice(df[[strategy]], p, start_date,  end_date)
            ret = period_performance(sliced[strategy])
            row_data[p] = f"{ret*100:.2f}%"

        # Full series within the chosen date window
        full_series = df[strategy].dropna()
        row_data["Sharpe"] = f"{annualized_sharpe_ratio(full_series):.2f}"
        row_data["Sortino"] = f"{annualized_sortino_ratio(full_series):.2f}"
        row_data["Max DD"] = f"{max_drawdown(full_series)*100:.2f}%"
        row_data["Avg Daily Return"] = f"{average_daily_return(full_series)*100:.4f}%"
        row_data["Hit Rate"] = f"{hit_rate(full_series)*100:.2f}%"
        
        table_rows.append(row_data)
    
    return table_rows




def Dashboard(df, start_date : str = None, end_date: str = None, plot_columns = None, table_columns = None):
    if plot_columns is None:
        plot_columns = df.columns
    if table_columns is None:
        table_columns = df.columns
    if end_date is None:
        end_date = pd.Timestamp(df.index[-1])
    if start_date is None:
        start_date = pd.Timestamp(df.index[0])

    app = dash.Dash(__name__)

    app.layout = html.Div([
       
        # Date Range Picker
        html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=start_date,
                end_date=end_date,
                display_format='YYYY-MM-DD'
            )
        ], style={'margin-bottom': '20px'}),

        # Dropdown for line plots and performance table strategies
        html.Div([
            html.Label("Select Strategies (Line Plots & Table):"),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[{'label': s, 'value': s} for s in df.columns],
                value=list(plot_columns),
                multi=True
            )
        ], style={'width': '400px', 'margin-bottom': '20px'}),

        # Cumulative Returns and Underwater plots
        html.Div([
            dcc.Graph(id='cumulative-returns-plot', style={'width': '100%'}),
            dcc.Graph(id='underwater-plot', style={'width': '100%'})
        ]),

        html.Hr(),

    # Period Returns Table
        html.Div([
            html.Div(id="date-range-label", style={"margin-bottom": "10px"}),
            html.H3("Performance Over Periods"),
            dash_table.DataTable(
                id='performance-period-table',
                columns=[
                    {"name": "Strategy", "id": "Strategy"}]
                    +
                    [{"name": p, "id": p, "type": "numeric", "format": Format(scheme=Scheme.percentage, precision=2)} for p in periods_columns],
            
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {'if': {'filter_query': '{WTD} >= 0', 'column_id': 'WTD'}, 'color': 'green'},
                    {'if': {'filter_query': '{WTD} < 0', 'column_id': 'WTD'}, 'color': 'red'},
                    {'if': {'filter_query': '{MTD} >= 0', 'column_id': 'MTD'}, 'color': 'green'},
                    {'if': {'filter_query': '{MTD} < 0', 'column_id': 'MTD'}, 'color': 'red'},
                    {'if': {'filter_query': '{YTD} >= 0', 'column_id': 'YTD'}, 'color': 'green'},
                    {'if': {'filter_query': '{YTD} < 0', 'column_id': 'YTD'}, 'color': 'red'},
                    {'if': {'filter_query': '{WTD} >= 0', 'column_id': 'WTD'}, 'color': 'green'},
                    {'if': {'filter_query': '{WTD} < 0', 'column_id': 'WTD'}, 'color': 'red'},
                    {'if': {'filter_query': '{MTD} >= 0', 'column_id': 'MTD'}, 'color': 'green'},
                    {'if': {'filter_query': '{MTD} < 0', 'column_id': 'MTD'}, 'color': 'red'},
                    {'if': {'filter_query': '{YTD} >= 0', 'column_id': 'YTD'}, 'color': 'green'},
                    {'if': {'filter_query': '{YTD} < 0', 'column_id': 'YTD'}, 'color': 'red'},
                    {'if': {'filter_query': '{ITD} >= 0', 'column_id': 'ITD'}, 'color': 'green'},
                    {'if': {'filter_query': '{ITD} < 0', 'column_id': 'ITD'}, 'color': 'red'},
                    {'if': {'filter_query': '{1M} >= 0', 'column_id': '1M'}, 'color': 'green'},
                    {'if': {'filter_query': '{1M} < 0', 'column_id': '1M'}, 'color': 'red'},
                    {'if': {'filter_query': '{3M} >= 0', 'column_id': '3M'}, 'color': 'green'},
                    {'if': {'filter_query': '{3M} < 0', 'column_id': '3M'}, 'color': 'red'},
                    {'if': {'filter_query': '{6M} >= 0', 'column_id': '6M'}, 'color': 'green'},
                    {'if': {'filter_query': '{6M} < 0', 'column_id': '6M'}, 'color': 'red'},
                    {'if': {'filter_query': '{1Y} >= 0', 'column_id': '1Y'}, 'color': 'green'},
                    {'if': {'filter_query': '{1Y} < 0', 'column_id': '1Y'}, 'color': 'red'},

                ]
            )
        ]),

        # Metrics Table
        html.Div([
            html.H3("Performance/Risk Metrics"),
            dash_table.DataTable(
                id='performance-metrics-table',
                columns=[
                    {"name": "Strategy", "id": "Strategy"},
                    {"name": "Sharpe", "id": "Sharpe"},
                    {"name": "Sortino", "id": "Sortino"},
                    {"name": "Max DD", "id": "Max DD"},
                    {"name": "Avg Daily Return", "id": "Avg Daily Return"},
                    {"name": "Hit Rate", "id": "Hit Rate"}
                ],
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            )
        ]),

        html.Hr(),
        # Dropdown for tables
        html.Div([
            html.Label("Select Strategies for Monthly and Weekly Tables:"),
            dcc.Dropdown(
                id='table-strategy-dropdown',
                options=[{'label': s, 'value': s} for s in df.columns],
                value=list(table_columns),
                multi=True
            )
        ], style={'width': '400px', 'margin-bottom': '20px'}),

        html.H3("Monthly Performance for Each Strategy"),
        html.Div(id='monthly-tables-container'),

        html.Hr(),

        html.H3("Weekly Performance (Last 15 Weeks)"),
        dash_table.DataTable(
            id='weekly-performance-table',
            columns=[],  # will be set dynamically
            data=[],     # will be set dynamically
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'},
            style_data_conditional=[]
        )
    ])

    # (Helper functions: compute_cumulative_returns, compute_drawdown, annualized_sharpe_ratio, 
    #  annualized_sortino_ratio, max_drawdown, average_daily_return, hit_rate, 
    #  period_performance, get_period_slice, compute_performance_table are assumed to be defined above.)


    @app.callback(
        [Output('weekly-performance-table', 'columns'),
        Output('weekly-performance-table', 'data'),
        Output('weekly-performance-table', 'style_data_conditional')],
        [Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('table-strategy-dropdown', 'value')]
    )
    def update_weekly_performance_table(start_date, end_date, selected_strategies):
        if not selected_strategies or not start_date or not end_date:
            return [], [], []

        # Convert input dates to Timestamps
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter DataFrame based on date range and selected strategies
        df_filtered = df.loc[(df.index >= start_date) & (df.index <= end_date), selected_strategies]

        # Resample to weekly returns on Mondays
        weekly_returns = (1 + df_filtered).resample('W-FRI').prod() - 1

        # Select the last 20 weeks
        last_15_weeks = weekly_returns.tail(15)

        # Transpose so rows are strategies, columns are weeks (Monday dates)
        transposed = last_15_weeks.transpose()

        # Format week start dates for column headers
        week_columns = [col.strftime('%Y-%m-%d') for col in last_15_weeks.index]

        # Prepare multi-index headers
        columns = [{"name": "Strategy", "id": "Strategy"}]
        week_ids = []  # to store unique column IDs
        for date in last_15_weeks.index:
            year = date.year
            month = month_names[date.month]
            # Count how many Mondays in the same month and year occurred up to this date in the dataset
            same_month_weeks = [d for d in last_15_weeks.index if d.year == year and d.month == date.month and d <= date]
            week_of_month = len(same_month_weeks)  # ordinal week in this month
            header = [str(year), month, f"Week {week_of_month}"]
            col_id = str(date)  # Use the date string as a unique column ID
            week_ids.append(col_id)

            columns.append({
                "name": header,
                "id": col_id,
                "type": "numeric",
                "format": Format(scheme=Scheme.percentage, precision=2)
            })

        # Prepare data for the table
        data = []
        for strat in transposed.index:
            row = {"Strategy": strat}
            for col_id, value in zip(week_ids, transposed.loc[strat].values):
                row[col_id] = value
            data.append(row)

        # Conditional styling: color text based on sign of returns
        style_data_conditional = []
        for col_id in week_ids:
            style_data_conditional += [
                {
                    'if': {
                        'filter_query': f'{{{col_id}}} >= 0',
                        'column_id': col_id
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': f'{{{col_id}}} < 0',
                        'column_id': col_id
                    },
                    'color': 'red'
                }
            ]

        return columns, data, style_data_conditional

    @app.callback(
        [Output('cumulative-returns-plot', 'figure'),
        Output('underwater-plot', 'figure'),
        Output('performance-period-table', 'data'),
        Output('performance-metrics-table', 'data'),
        Output('date-range-label', 'children')],
        [Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('strategy-dropdown', 'value')]
    )

    def update_dashboard(start_date, end_date, selected_strategies):
        """
        Updates:
        1) Cumulative Returns (multi-line)
        2) Underwater (multi-line)
        3) Performance Table
        based on the date range and selected strategies.
        """
        if not selected_strategies or not start_date or not end_date:
            empty_fig = go.Figure(data=[])
            return empty_fig, empty_fig, []

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter data
        df_filtered = df.loc[(df.index >= start_date) & 
                                    (df.index <= end_date),
                                    selected_strategies]

        if df_filtered.empty:
            empty_fig = go.Figure(data=[])
            return empty_fig, empty_fig, []

        # 1) Cumulative Returns (multi-line)
        cum_returns_fig = go.Figure()
        for strat in df_filtered.columns:
            strat_returns = df_filtered[strat].dropna()
            cum_ret = (1 + strat_returns).cumprod() - 1
            cum_returns_fig.add_trace(go.Scatter(
                x=cum_ret.index,
                y=cum_ret,
                mode='lines',
                name=strat,
                line=dict(width=1) 
            ))
        cum_returns_fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".2%",
            hovermode='x unified'
        )

        # 2) Underwater Plot (multi-line)
        underwater_fig = go.Figure()
        for strat in df_filtered.columns:
            strat_returns = df_filtered[strat].dropna()
            cum_ret = (1 + strat_returns).cumprod()
            dd = compute_drawdown(cum_ret)
            underwater_fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd,
                mode='lines',
                name=strat,
                line=dict(width=1) 
            ))
        underwater_fig.update_layout(
            title="Underwater Plot (Drawdown)",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".2%",
            hovermode='x unified',

        )

        # Compute full performance table data
        table_data = compute_performance_table(df.loc[(df.index >= start_date) & 
                                    (df.index <= end_date)], start_date, end_date)

    
        for row in table_data:
            for col in periods_columns:
                val = row.get(col, "")
                try:
                    row[col] = float(val.strip('%'))/100 if isinstance(val, str) else val
                except:
                    row[col] = None

        # Separate data for period and metrics tables
        metrics_cols = {"Strategy", "Sharpe", "Sortino", "Max DD", "Avg Daily Return", "Hit Rate"}
        period_data = []
        metrics_data = []
        for row in table_data:
            period_row = {k: row.get(k, None) for k in ["Strategy"] + periods_columns}
            metrics_row = {k: row.get(k, "") for k in metrics_cols}
            period_data.append(period_row)
            metrics_data.append(metrics_row)
        
        
        # 4) Prepare the date-range text
        date_text = f"Start: {start_date.date()}  |  End: {end_date.date()}"

        return cum_returns_fig, underwater_fig, period_data, metrics_data, date_text

    @app.callback(
        Output('monthly-tables-container', 'children'),
        [Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('table-strategy-dropdown', 'value')]
    )
    def update_monthly_tables(start_date, end_date, selected_strategies):
        if not selected_strategies:
            return []

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        tables = []

        for strat in selected_strategies:
                        # Filter data and compute monthly returns
            filtered_df = df[(df.index >= start_date) & (df.index <= end_date)][[strat]]
            monthly_returns = (1 + filtered_df).resample('M').prod() - 1
            monthly_returns['Year'] = monthly_returns.index.year
            monthly_returns['Month'] = monthly_returns.index.month

            # Pivot the table so that each year is a row and months (1-12) are columns
            pivot = monthly_returns.pivot(index='Year', columns='Month', values=strat)
            pivot = pivot.sort_index(ascending=False)

            # Compute YTD performance:
            # First, reindex to ensure all months from 1 to 12 exist, fill missing months with 0 (0% return)
            pivot_months = pivot.reindex(columns=range(1, 13), fill_value=0)
            # Calculate YTD as the cumulative product over the months minus 1
            pivot['Y'] = (pivot_months + 1).prod(axis=1) - 1

            # Define column order with YTD at the right
            all_months = list(range(1, 13)) + ["Y"]
            # Ensure the index covers the full range of years
            all_years = reversed(range(pivot.index.min(), pivot.index.max() + 1)) if not pivot.empty else []
            pivot = pivot.reindex(index=all_years, columns=all_months, fill_value=np.nan)

            # Rename month columns using your month_names mapping (YTD remains unchanged)
            renamed_columns = {m: month_names.get(m, m) for m in range(1, 13)}
            renamed_columns["Y"] = "Y"
            pivot.rename(columns=renamed_columns, inplace=True)
            pivot.reset_index(inplace=True)

            # Build columns configuration for your table component
            columns = []
            for col in pivot.columns:
                if col == 'Year':
                    columns.append({"name": col, "id": col})
                else:
                    columns.append({
                        "name": col,
                        "id": col,
                        "type": "numeric",
                        "format": Format(scheme=Scheme.percentage, precision=1)
                    })

            # Convert pivot table to dictionary format for your table
            data = pivot.to_dict('records')

            # Conditional styling based on performance (green for >=0 and red for <0)
            style_data_conditional = []
            for month in month_names.values():
                style_data_conditional += [
                    {
                        'if': {
                            'filter_query': f'{{{month}}} >= 0',
                            'column_id': month
                        },
                        'color': 'green'
                    },
                    {
                        'if': {
                            'filter_query': f'{{{month}}} < 0',
                            'column_id': month
                        },
                        'color': 'red'
                    }
                ]
            # Optionally add styling for the YTD column if needed:
            style_data_conditional += [
                {
                    'if': {
                        'filter_query': '{Y} >= 0',
                        'column_id': 'Y'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': '{Y} < 0',
                        'column_id': 'Y'
                    },
                    'color': 'red'
                }
            ]

            table = dash_table.DataTable(
                columns=columns,
                data=data,
                style_data_conditional=style_data_conditional,
                style_table={'overflowX': 'auto', 'margin': '10px 0'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                page_action='none'
            )

            tables.append(html.H4(strat))
            tables.append(table)

        return tables
    
    app.run_server(debug=True)




# Sample DataFrame generation for demonstration
dates = pd.date_range("2024-07-01", periods=500, freq="B")
np.random.seed(42)
df = pd.DataFrame(np.random.normal(0.001, 0.02, size=(500, 3)), 
                  index=dates, 
                  columns=["Strategy_A", "Strategy_B", "Strategy_C"])

Dashboard(df, plot_columns=["Strategy_A"])