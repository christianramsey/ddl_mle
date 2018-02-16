import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.lib.io import file_io
tf.logging.set_verbosity(tf.logging.INFO)
from pprint import pprint 

# get the data
# load training and eval files    
traindir = 'data/train/*'
evaldir = 'data/test/*'

# traindata =   [file for file in file_io.get_matching_files(traindir)]
# evaldata =    [file for file in file_io.get_matching_files(evaldir)]

# DESCRIBE DATASET
# define columns and field defaults
COLUMNS        = ["Lat", "Long", "Altitude","Date_",
                  "Time_", "dt_", "y"]
FIELD_DEFAULTS = [[0.], [0.], [0.], ['na'],
                  ['na'], ['na'], ['na']]
feature_names = COLUMNS[:-1]

# FEATURE COLUMNS
## represent feature columns
# dense feature_columns
lat      = tf.feature_column.numeric_column("Lat")
lng      = tf.feature_column.numeric_column("Long")
altitude = tf.feature_column.numeric_column("Altitude")

# sparse feature_columns
date_ = tf.feature_column.categorical_column_with_hash_bucket('Date_', 3650)
time_ = tf.feature_column.categorical_column_with_hash_bucket('Time_', 10000)
dt_ = tf.feature_column.categorical_column_with_hash_bucket('dt_', 10000)

lat_long_buckets = list(np.linspace(-180.0, 180.0, num=1000))

lat_buck  = tf.feature_column.bucketized_column(
    source_column = lat,
    boundaries = lat_long_buckets )

lng_buck = tf.feature_column.bucketized_column(
    source_column = lng,
    boundaries = lat_long_buckets)

real_fc  = [lat, lng, altitude]
sparse_fc =  [date_, time_, dt_, lat_buck, lng_buck ]



