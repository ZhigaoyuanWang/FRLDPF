import csv
import numpy as np

def read_csv(filename):
    cities = set()
    jobs = set()
    dates = []
    data = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            city, job, date, value = row
            cities.add(city)
            jobs.add(job)
            dates.append(date)
            data.append([city, job, date, float(value)])

    # Sort dates
    dates = sorted(set(dates))

    # Create mappings for city and job
    city_to_index = {city: i for i, city in enumerate(sorted(cities))}
    job_to_index = {job: i for i, job in enumerate(sorted(jobs))}
    date_to_index = {date: i for i, date in enumerate(dates)}

    # Create 3D array
    C = len(cities)
    J = len(jobs)
    T = len(dates)
    array = np.zeros((C, J, T))

    for city, job, date, value in data:
        array[city_to_index[city], job_to_index[job], date_to_index[date]] = value

    return array, sorted(cities), sorted(jobs), dates

# Example usage
array, cities, jobs, dates = read_csv('spatial_temporal_data.csv')
print("Array shape:", array.shape)
print("Cities:", cities)
print("Jobs:", jobs)
print("Dates:", dates)
np.save('labor_demand.npy',array)
