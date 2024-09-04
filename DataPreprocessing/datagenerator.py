import csv
import random
import pandas as pd

start_date = '2017-01-01'
dates = pd.date_range(start=start_date, periods=378, freq='W')
dates_list = [date.strftime('%Y-%m-%d') for date in dates]


cities = list(range(15))
job_positions = list(range(144))
data_range = 100
def generate_time_series_csv(filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for c in cities:
            for j in job_positions:
                for t in dates_list:
                    v = random.randint(1, data_range)
                    writer.writerow([c ,j ,t, v])

# Example usage:
generate_time_series_csv("spatial_temporal_data.csv")
