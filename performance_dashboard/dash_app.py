import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt



def Dashboard(df: pd.DataFrame, start_date: str):
    """
    Launch the Dash performance dashboard given a returns DataFrame.
    
    :param df: Pandas DataFrame of daily returns (index=dates, columns=strategies).
    :param port: Optional port to run the server on (defaults to 8050).
    """

    start_date = pd.Timestamp(start_date)

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
        return (returns_series >= 0).mean()

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
        elif period == "SINCE INCEPTION":
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
        else:
            start = global_start

        return df.loc[(df.index >= start) & (df.index <= actual_end_date)]

    def compute_performance_table(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        For each strategy (column in df), compute:
        - WTD, MTD, YTD, Since Inception,
        - 1M, 3M, 6M,
        - Sharpe, Sortino, MaxDD, AvgDaily, HitRate,
        all restricted to the filtered data [start_date, end_date].
        """
        if df.empty:
            return []

        periods = ["WTD", "MTD", "YTD", "SINCE INCEPTION", "1M", "3M", "6M"]
        table_rows = []
        
        for strategy in df.columns:
            row_data = {"Strategy": strategy}
            for p in periods:
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


    # ------------------------------------------------------------------------------
    # DASH APP
    # ------------------------------------------------------------------------------
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H2("Strategy Performance Dashboard"),

        # --------------------------
        # Date Range + Strategy selection FOR LINE PLOTS & TABLE
        # --------------------------
        html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date = start_date,
                end_date=df.index[-1],
                display_format='YYYY-MM-DD'
            )
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Label("Select Strategies (Line Plots & Table):"),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[{'label': s, 'value': s} for s in df.columns],
                value=list(df.columns),  # default: all
                multi=True
            )
        ], style={'width': '400px', 'margin-bottom': '20px'}),

        # --------------------------
        # Cumulative returns + Underwater
        # --------------------------
        html.Div([
        dcc.Graph(
            id='cumulative-returns-plot',
            style={'width': '100%'}
        ),
        dcc.Graph(
            id='underwater-plot',
            style={'width': '100%'}
        )
        ]),

        html.Hr(),

        # --------------------------
        # Performance table
        # --------------------------
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
        ]),

        html.Hr(),

        # --------------------------
        # HEATMAPS (separate dropdown so it doesn't interfere)
        # --------------------------
        html.Div([
            html.Label("Select Strategies (Heatmaps):"),
            dcc.Dropdown(
                id='heatmap-strategy-dropdown',
                options=[{'label': s, 'value': s} for s in df.columns],
                value=list(df.columns),  # default: all
                multi=True
            )
        ], style={'width': '400px', 'margin-bottom': '20px'}),

        html.Div([
            dcc.Graph(id='monthly-heatmap'),
            dcc.Graph(id='weekly-heatmap')
        ])
    ])

    # ------------------------------------------------------------------------------
    # CALLBACK: LINE PLOTS + TABLE
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
                name=strat
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
        table_data = compute_performance_table(df_filtered, start_date, end_date)

        return cum_returns_fig, underwater_fig, table_data

    # ------------------------------------------------------------------------------
    # CALLBACK: HEATMAPS (MONTHLY & WEEKLY)
    # ------------------------------------------------------------------------------
    @app.callback(
        [
            Output('monthly-heatmap', 'figure'),
            Output('weekly-heatmap', 'figure')
        ],
        [Input('heatmap-strategy-dropdown', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')]
    )
    def update_heatmaps(selected_heatmap_strategies, start_date, end_date):
        """
        Creates:
        - A multi-subplot monthly heatmap
        - A multi-subplot weekly heatmap
        for the chosen strategies. (Uses entire df, ignoring the date range.)
        """
        if not selected_heatmap_strategies:
            empty_fig = go.Figure()
            return empty_fig, empty_fig

        # ---------- MONTHLY HEATMAP -----------
        # Resample to monthly returns
        monthly_returns = (1 + df[(df.index >= start_date) & (df.index <= end_date)]).resample('M').prod() - 1
        # We'll create a separate subplot for each strategy
        rows_m = len(selected_heatmap_strategies)
        fig_monthly = make_subplots(
            rows=rows_m, cols=1,
            subplot_titles=[f"{s} - Monthly Returns" for s in selected_heatmap_strategies],
            vertical_spacing=0.1
        )

        for i, strat in enumerate(selected_heatmap_strategies):
            # Extract that strategy's monthly
            strat_monthly = monthly_returns[[strat]].copy()
            if strat_monthly.empty:
                continue

            # Add year & month columns for pivoting
            strat_monthly['Year'] = strat_monthly.index.year
            strat_monthly['Month'] = strat_monthly.index.month

            # Pivot: rows=Year, cols=Month, values=returns
            pivot = strat_monthly.pivot(index='Year', columns='Month', values=strat)
            pivot = pivot.sort_index(ascending=False)  # so most recent year at top (optional)

            # We'll create a heatmap
            heatmap = go.Heatmap(
                x=pivot.columns,
                y=pivot.index,
                z=pivot.values ,
                colorscale='rdylgn',
                hovertemplate="Year=%{y}<br>Month=%{x}<br>Return=%{z:.2%}<extra></extra>",
                texttemplate="%{z:.1%}",
                showscale=False
                
                
            )
            fig_monthly.add_trace(heatmap, row=i+1, col=1)

            # Format axes
            fig_monthly.update_xaxes(
                tickmode='array',
                tickvals=list(range(1,13)),
                ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                title_text="Month",
                row=i+1, col=1,
                
            )
            fig_monthly.update_yaxes(title_text="Year", row=i+1, col=1)

        fig_monthly.update_layout(
            height=300*rows_m,  # scale figure height by number of subplots
            title="Monthly Returns Heatmap(s)"
        )

        # ---------- WEEKLY HEATMAP -----------
        # Resample to weekly returns
        # We'll say 'W-MON' means each period ends on Monday
        weekly_returns = (1 + df[(df.index >= start_date) & (df.index <= end_date)]).resample('W-MON').prod() - 1
        rows_w = len(selected_heatmap_strategies)
        fig_weekly = make_subplots(
            rows=rows_w, cols=1,
            subplot_titles=[f"{s} - Weekly Returns" for s in selected_heatmap_strategies],
            vertical_spacing=0.1
        )

        for i, strat in enumerate(selected_heatmap_strategies):
            strat_weekly = weekly_returns[[strat]].copy()
            if strat_weekly.empty:
                continue

            # Add Year & Week for pivoting
            # isocalendar().week gives the ISO week number
            strat_weekly['Year'] = strat_weekly.index.year
            strat_weekly['Week'] = strat_weekly.index.isocalendar().week

            # Pivot: rows=Year, cols=Week
            pivot = strat_weekly.pivot(index='Year', columns='Week', values=strat)
            pivot = pivot.sort_index(ascending=False)  # so the latest year is on top

            heatmap = go.Heatmap(
                x=pivot.columns,   # weeks of the year
                y=pivot.index,     # year
                z=pivot.values,
                colorscale='rdylgn',
                hovertemplate="Year=%{y}<br>Week=%{x}<br>Return=%{z:.2%}<extra></extra>",
                texttemplate="%{z:.1%}",
                showscale=False
            )
            fig_weekly.add_trace(heatmap, row=i+1, col=1)

            fig_weekly.update_xaxes(title_text="Week #", row=i+1, col=1)
            fig_weekly.update_yaxes(title_text="Year", row=i+1, col=1)

        fig_weekly.update_layout(
            height=300*rows_w,
            title="Weekly Returns Heatmap(s)"
        )

        return fig_monthly, fig_weekly


    app.run_server(debug=True)

# Suppose you have a DataFrame 'df'
dates = pd.date_range("2021-01-01", periods=500, freq="B")
df = pd.DataFrame(np.random.normal(0.001, 0.01, size=(500, 2)), 
                  index=dates, columns=["Strategy_X", "Strategy_Y"])

# Simply call Dashboard
Dashboard(df, start_date = "2021-07-01")
