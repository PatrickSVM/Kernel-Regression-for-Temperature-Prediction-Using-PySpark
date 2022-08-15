from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
import datetime
from pyspark import SparkContext

########################################################################################################################
#################################################### CONFIGURATION #####################################################
########################################################################################################################

h_distance = 100 # Up to you (km)

h_date = 20 # Up to you (days)

h_time = 180 # Up to you (minutes)

a = 58.4274 # Up to you (latetude)

b = 14.826 # Up to you (longitude)

date = "2013-07-04" # Up to you (format "%Y-%m-%d")

########################################################################################################################
#################################################### IMPLEMENTATION ####################################################
########################################################################################################################

def parse_time(input):
  """
  Parse an input string of the format "%H:%M:%S" to a datetime object.
  """
  input_split = input.split(':')
  if input_split[0] == '24':
    return datetime.datetime.strptime("00:" + input_split[1] + ":" + input_split[2], "%H:%M:%S")
  return datetime.datetime.strptime(input, "%H:%M:%S")

def gauss(x):
  """
  Calculates the y value for a given x of the gauss function
  """
  return exp(-x**2)

def haversine(lon1, lat1, lon2, lat2):
  """
  Calculate the great circle distance between two points
  on the earth (specified in decimal degrees)
  """
  # convert decimal degrees to radians
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
  # haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * asin(sqrt(a))
  km = 6367 * c
  return km

def kernel_distance(lat1, lon1, lat2, lon2, h_distance):
  """
  Calculates the value of the distance between two points which is 
  handed over to an activation function (in this case the gauss function)
  """
  return gauss(haversine(lon1, lat1, lon2, lat2)/h_distance)

def distance_dates(date1, date2):
  """
  Calculates the distance in days between two datetime objects
  """
  return date1.timetuple().tm_yday - date2.timetuple().tm_yday

def kernel_date(date1, date2, h_date):
  """
  Calculates the value of the distance between two dates which is 
  handed over to an activation function (in this case the gauss function)
  """
  return gauss(distance_dates(date1, date2)/h_date)

def distance_time(time1, time2):
  """
  Calculates the distance in minutes between two datetime objects.
  """
  return (time1 - time2).seconds // 60

def kernel_time(time1, time2, h_time):
  """
  Calculates the value of the distance between two time points which is 
  handed over to an activation function (in this case the gauss function)
  """
  return gauss(distance_time(time1, time2)/h_time)

########################################################################################################################
######################################################## "MAIN" ########################################################
########################################################################################################################

# Set the spark context
sc = SparkContext(appName = "lab_kernel")

# Load the data from the hdfs
temperature_file = sc.textFile("BDA/input/temperature-readings.csv")
temperature_file_lines = temperature_file.map(lambda line: line.split(";"))

stations_file = sc.textFile("BDA/input/stations.csv")
stations_file_lines = stations_file.map(lambda line: line.split(";"))

# Map station id to their kernel distance to a and b
stations_distance = stations_file_lines \
  .map(lambda line: (line[0], kernel_distance(float(line[3]), float(line[4]), a , b, h_distance))) \
  .collectAsMap()
sc.broadcast(stations_distance)

# "Join" temperatures with distance from a and b to the temperatures station
# Datastructure of result: stationId, date, time, temperature, kernel distance from station to a and b, date difference
# Filter by date to just use data from the past and today -> Just filter for the day
date_parsed = datetime.datetime.strptime(date, "%Y-%m-%d")
temperatures_station_distances = temperature_file_lines \
  .map(lambda line: (line[0], 
    datetime.datetime.strptime(line[1], "%Y-%m-%d"), 
    parse_time(line[2]), float(line[3]), 
    stations_distance[line[0]], 
    (date_parsed - datetime.datetime.strptime(line[1], "%Y-%m-%d")).days)) \
  .filter(lambda line: line[5] >= 0)

# Calculate kernel distance between day of temperature and day we want to predict
# Datastructure of result: kernel_distance_to_station, kernel_distance_to_date, time, temperature, date difference
temperatures_station_distances_days = temperatures_station_distances \
  .map(lambda line: (line[4], kernel_date(date_parsed, line[1], h_date), line[2], line[3], line[5]))
temperatures_station_distances_days.cache()


for time in ["24:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00",
"12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]:

  time_parsed = parse_time(time)
  # Datastructure of result: kernel_distance_to_station, kernel_distance_to_date, kernel_distance_time, temperature
  # Filter all values before our set time at the set day
  # Cache the result
  temperatures_station_distances_days_time = temperatures_station_distances_days \
    .filter(lambda line: (line[4] > 0) or (line[4] == 0 and (time_parsed - line[2]).seconds) > 0) \
    .map(lambda line: (line[0], line[1], kernel_time(time_parsed, line[2], h_time), line[3]))

  temperatures_station_distances_days_time.cache()

  # Create new column for sum and product of all kernels for each row
  temperatures_kernel_sum_multi = temperatures_station_distances_days_time \
    .map(lambda line: (line[0] + line[1] + line[2], line[0] * line[1] * line[2], line[3]))

  temperatures_kernel_sum_multi.cache()

  # Calculate the sum of all kernels 
  # First: For summed kernel
  temperatures_kernel_sum_total = temperatures_kernel_sum_multi \
    .map(lambda line: ("1", line[0])) \
    .reduceByKey(lambda a,b: a+b) \
    .collect()[0][1]

  # Second: For multiplied kernel
  temperatures_kernel_prod_total = temperatures_kernel_sum_multi \
    .map(lambda line: ("1", line[1])) \
    .reduceByKey(lambda a,b: a + b) \
    .collect()[0][1]


  # Calculate the product of each temperature and its sum of kernel
  product_temperatures_kernel_sum_total = temperatures_kernel_sum_multi \
    .map(lambda line: line[0] * line[2]) \
    .reduce(lambda a,b: a + b)

  product_temperatures_kernel_prod_total = temperatures_kernel_sum_multi \
    .map(lambda line: line[1] * line[2]) \
    .reduce(lambda a,b: a + b)


  # Divide the product of each temperature and its sum of kernel through the sum of all kernels
  result_sum = product_temperatures_kernel_sum_total / temperatures_kernel_sum_total
  result_prod = product_temperatures_kernel_prod_total / temperatures_kernel_prod_total


  print(time, result_sum, result_prod)
