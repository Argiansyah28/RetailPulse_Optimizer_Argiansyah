import pandas as pd
import numpy as np
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=730)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

n = len(dates)
trend = np.linspace(50, 150, n)
seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.normal(0, 10, n)

sales = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'sales': sales.astype(int)})
df.to_csv('retail_sales.csv', index=False)