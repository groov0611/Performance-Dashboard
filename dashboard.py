import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt

# ------------------------------------------------------------------------------
# SAMPLE DATA: Create a fake returns DataFrame for demonstration.
# In your actual code, replace this with your real returns data.
# ------------------------------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=750, freq="B")  # 750 business days
strategies = ["Strategy_A", "Strategy_B", "Strategy_C"]
data = np.random.normal(loc=0.0005, scale=0.01, size=(750, len(strategies)))
returns_df = pd.DataFrame(data, index=dates, columns=strategies)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def compute_cumulative_returns(returns_series: pd.Series) -> pd.Series:
    """
    Given a daily returns series, compute the cumulative returns over time.
    """
    return (1 + returns_series).cumprod() - 1

def compute_drawdown(cum_returns: pd.Series) -> pd.Series:
    """
    Given a cumulative returns series, compute drawdown at each point in time.
    Drawdown(t) = 1 - (cum_returns(t) / cum_returns[:t].max())
    """
    running_max = cum_returns.cummax()
    drawdown = 1 - (cum_returns / running_max)
    return drawdown

def annualized_sharpe_ratio(returns_series: pd.Series, freq=252) -> float:
    """
    Annualized Sharpe Ratio = (mean(daily_returns) / std(daily_returns)) * sqrt(freq)
    """
    if returns_series.std() == 0:
        return 0
    return (returns_series.mean() / returns_series.std()) * np.sqrt(freq)

def annualized_sortino_ratio(returns_series: pd.Series, freq=252) -> float:
    """
    Sortino Ratio = (mean(daily_returns) / std(negative_daily_returns)) * sqrt(freq)
    Only downside (negative) returns are considered in the denominator.
    """
    negative_returns = returns_series[returns_series < 0]
    if negative_returns.std() == 0:
        return 0
    return (returns_series.mean() / negative_returns.std()) * np.sqrt(freq)

def max_drawdown(returns_series: pd.Series) -> float:
    """
    Compute maximum drawdown from a daily returns series.
    """
    cum = (1 + returns_series).cumprod()
    dd = compute_drawdown(cum)
    return dd.max()

def average_daily_return(returns_series: pd.Series) -> float:
    """
    Average daily return
    """
    return returns_series.mean()

def hit_rate(returns_series: pd.Series) -> float:
    """
    Percentage of days with non-negative returns (counting zero as positive).
    """
    return (returns_series >= 0).mean()

def period_performance(returns_series: pd.Series) -> float:
    """
    Given daily returns for a period, compute the total return
    for that period.  i.e. (1 + r_1)*(1 + r_2)*...*(1 + r_n) - 1
    """
    if len(returns_series) == 0:
        return 0
    return (1 + returns_series).prod() - 1

def get_period_slice(df: pd.DataFrame, period: str, end_date: pd.Timestamp):
    """
    Helper to slice the (already date-filtered) DataFrame index to get WTD, MTD, YTD, etc.
    We'll interpret 'today' as the 'end_date' within the chosen date range.
    """
    if df.empty:
        return df

    # Because the user might pick an end_date beyond the actual data index,
    # let's find the actual last date in df that is <= end_date.
    valid_dates = df.index[df.index <= end_date]
    if len(valid_dates) == 0:
        return df.iloc[0:0]  # empty
    actual_end_date = valid_dates[-1]

    # For sub-periods, define:
    start_of_week = actual_end_date - pd.to_timedelta(actual_end_date.weekday(), unit='D')
    start_of_month = pd.to_datetime(f"{actual_end_date.year}-{actual_end_date.month:02d}-01")
    start_of_year = pd.to_datetime(f"{actual_end_date.year}-01-01")

    # We'll restrict ourselves to the portion of df up to 'actual_end_date'.
    # Then for each subperiod, we slice from the appropriate start date to that end date.
    # But also ensure we don't go before the global start_date in the overall filter.
    global_start = df.index.min()
    if period == "WTD":
        start = max(global_start, start_of_week)
    elif period == "MTD":
        start = max(global_start, start_of_month)
    elif period == "YTD":
        start = max(global_start, start_of_year)
    elif period == "SINCE INCEPTION":
        start = global_start
    elif period == "1M":  # Rolling 1 month ~ last 21 business days
        # We'll just pick last 21 rows up to actual_end_date
        # (Though you could do something more precise with calendar months.)
        idx = df.loc[:actual_end_date].index
        start_idx = max(0, len(idx) - 21)
        start = idx[start_idx]
    elif period == "3M":  # Rolling 3 months ~ last 63 business days
        idx = df.loc[:actual_end_date].index
        start_idx = max(0, len(idx) - 63)
        start = idx[start_idx]
    elif period == "6M":  # Rolling 6 months ~ last 126 business days
        idx = df.loc[:actual_end_date].index
        start_idx = max(0, len(idx) - 126)
        start = idx[start_idx]
    else:
        start = global_start

    return df.loc[(df.index >= start) & (df.index <= actual_end_date)]

def compute_performance_table(df: pd.DataFrame, end_date: pd.Timestamp):
    """
    For each strategy (column in df), compute:
      - WTD, MTD, YTD, SINCE INCEPTION,
      - 1M, 3M, 6M,
      - Sharpe, Sortino, Max Drawdown, Avg Daily Return, Hit Rate
    restricted to the user-chosen date range (df is already date-filtered externally).
    'end_date' is the user-chosen end_date, used for subperiod slicing logic.
    """
    if df.empty:
        return []

    periods = ["WTD", "MTD", "YTD", "SINCE INCEPTION", "1M", "3M", "6M"]
    table_rows = []
    
    for strategy in df.columns:
        row_data = {"Strategy": strategy}
        for p in periods:
            sliced = get_period_slice(df[[strategy]], p, end_date)
            ret = period_performance(sliced[strategy])
            row_data[p] = f"{ret*100:.2f}%"

        # Full series *within the chosen date window* for ratio calculations
        full_series = df[strategy].dropna()
        row_data["Sharpe"] = f"{annualized_sharpe_ratio(full_series):.2f}"
        row_data["Sortino"] = f"{annualized_sortino_ratio(full_series):.2f}"
        row_data["Max DD"] = f"{max_drawdown(full_series)*100:.2f}%"
        row_data["Avg Daily Return"] = f"{average_daily_return(full_series)*100:.4f}%"
        row_data["Hit Rate"] = f"{hit_rate(full_series)*100:.2f}%"
        
        table_rows.append(row_data)
    
    return table_rows

# ------------------------------------------------------------------------------
# DASH APP
# ------------------------------------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Strategy Performance Dashboard"),

    # Date range picker
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=returns_df.index[0],       # default: from earliest date
            end_date=returns_df.index[-1],        # default: through latest date
            display_format='YYYY-MM-DD'
        )
    ], style={'margin-bottom': '20px'}),

    # Multi-select strategies
    html.Div([
        html.Label("Select Strategies:"),
        dcc.Dropdown(
            id='strategy-dropdown',
            options=[{'label': s, 'value': s} for s in returns_df.columns],
            value=list(returns_df.columns),  # default: all
            multi=True
        )
    ], style={'width': '400px', 'margin-bottom': '20px'}),

    # Graphs
    html.Div([
        dcc.Graph(id='cumulative-returns-plot'),
        dcc.Graph(id='underwater-plot')
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Hr(),

    # Performance table
    html.Div([
        html.H3("Performance Table"),
        dash_table.DataTable(
            id='performance-table',
            columns=[
                {"name": "Strategy", "id": "Strategy"},
                {"name": "WTD", "id": "WTD"},
                {"name": "MTD", "id": "MTD"},
                {"name": "YTD", "id": "YTD"},
                {"name": "SINCE INCEPTION", "id": "SINCE INCEPTION"},
                {"name": "1M (Rolling)", "id": "1M"},
                {"name": "3M (Rolling)", "id": "3M"},
                {"name": "6M (Rolling)", "id": "6M"},
                {"name": "Sharpe", "id": "Sharpe"},
                {"name": "Sortino", "id": "Sortino"},
                {"name": "Max DD", "id": "Max DD"},
                {"name": "Avg Daily Return", "id": "Avg Daily Return"},
                {"name": "Hit Rate", "id": "Hit Rate"},
            ],
            data=[],  # updated by callback
            style_table={'overflowX': 'auto'},
        )
    ])
])

# ------------------------------------------------------------------------------
# CALLBACK
# ------------------------------------------------------------------------------
@app.callback(
    [
        Output('cumulative-returns-plot', 'figure'),
        Output('underwater-plot', 'figure'),
        Output('performance-table', 'data')
    ],
    [
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('strategy-dropdown', 'value')
    ]
)
def update_dashboard(start_date, end_date, selected_strategies):
    """
    1) Filter returns_df by the chosen date range and selected strategies
    2) Compute multi-line cumulative returns and underwater plots
    3) Compute table stats for the filtered data
    """

    if not selected_strategies or start_date is None or end_date is None:
        # Return empty figs / table if no selections
        empty_fig = go.Figure(data=[])
        return empty_fig, empty_fig, []

    # Convert start/end_date to Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame by date range and strategies
    df_filtered = returns_df.loc[(returns_df.index >= start_date) & 
                                 (returns_df.index <= end_date),
                                 selected_strategies]

    # If filtered data is empty, return empty outputs
    if df_filtered.empty:
        empty_fig = go.Figure(data=[])
        return empty_fig, empty_fig, []

    # 1) Cumulative Returns Plot - multiple lines
    cum_returns_fig = go.Figure()
    for strat in df_filtered.columns:
        strat_returns = df_filtered[strat].dropna()
        if len(strat_returns) == 0:
            continue
        cum_ret = (1 + strat_returns).cumprod() - 1
        cum_returns_fig.add_trace(go.Scatter(
            x=cum_ret.index,
            y=cum_ret,
            mode='lines',
            name=strat
        ))
    cum_returns_fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".2%",
        hovermode='x unified'
    )

    # 2) Underwater (drawdown) Plot - multiple lines
    underwater_fig = go.Figure()
    for strat in df_filtered.columns:
        strat_returns = df_filtered[strat].dropna()
        if len(strat_returns) == 0:
            continue
        cum_ret = (1 + strat_returns).cumprod()
        dd = compute_drawdown(cum_ret)
        underwater_fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd,
            mode='lines',
            name=strat
        ))
    underwater_fig.update_layout(
        title="Underwater Plot (Drawdown)",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".2%",
        hovermode='x unified'
    )

    # 3) Performance Table
    # We pass the entire filtered df (all strategies), and the chosen end_date
    table_data = compute_performance_table(df_filtered, end_date)

    return cum_returns_fig, underwater_fig, table_data

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
