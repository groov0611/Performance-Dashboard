# Performance Dashboard

A Python package providing a Dash-based interactive dashboard to monitor strategy performance. This package includes:

- Cumulative returns plot  
- Underwater (drawdown) plot  
- Performance table (WTD, MTD, YTD, etc.)  
- Monthly and weekly returns heatmaps  

## Installation

Install directly from your Git repository (replace the example URL with your own):

```bash
pip install git+https://github.com/YourUser/performance-dashboard.git
```



```python
import pandas as pd
import numpy as np
from performance_dashboard import Dashboard

# Example: synthetic returns DataFrame
dates = pd.date_range("2021-01-01", periods=100, freq="B")
df = pd.DataFrame(
    np.random.normal(0.001, 0.01, size=(100, 2)),
    index=dates,
    columns=["Strategy_X", "Strategy_Y"]
)

# Launch the Dash app
Dashboard(df)  # By default, runs on http://127.0.0.1:8050
```