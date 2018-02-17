import glob, os
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.lib.io import file_io
tf.logging.set_verbosity(tf.logging.INFO)

# define columns and field defaults
COLUMNS        = ["Lat", "Long", "Altitude","Date_",
                  "Time_", "dt_", "y"]
FIELD_DEFAULTS = [[0.], [0.], [0.], ['na'],
                  ['na'], ['na'], ['na']]
feature_names = COLUMNS[:-1]

# define input pipeline
def my_input_fn(file_paths, perform_shuffle=True, repeat_count=10000,  batch_size=32):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, FIELD_DEFAULTS)
        label = tf.convert_to_tensor(parsed_line[-1:])
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything (but last element) are the features
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_paths)  # Read text file
                    .skip(1)  # Skip header row
                    .map(decode_csv))  # Transform each elem by decode_csv
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)    

    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

# define serving input for predictions
def serving_input_fn():
    feature_placeholders = {
        'Lat': tf.placeholder(tf.float32, [None]),
        'Long': tf.placeholder(tf.float32, [None]),
        'Altitude': tf.placeholder(tf.float32, [None]),
        'Date_': tf.placeholder(tf.string, [None]),
        'Time_': tf.placeholder(tf.string, [None]),
        'dt_': tf.placeholder(tf.string, [None]),
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

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

# feature engineering (embeddings and feature crosses)
crossed_lat_lon = tf.feature_column.crossed_column(
    [lat_buck, lng_buck], 7000)

lng_buck_embedding = tf.feature_column.embedding_column(
    categorical_column=lng_buck,
    dimension=3)

lat_buck_embedding = tf.feature_column.embedding_column(
    categorical_column=lat_buck,
    dimension=3)

crossed_ll_embedding = tf.feature_column.embedding_column(
    categorical_column=crossed_lat_lon,
    dimension=12)

crossed_all = tf.feature_column.crossed_column(
    ['Lat', 'Long', 'Date_', 'Time_', 'dt_'], 20000)

crossed_all_embedding = tf.feature_column.embedding_column(
    categorical_column=crossed_all,
    dimension=89)

date_embedding = tf.feature_column.embedding_column(
    categorical_column=date_,
    dimension=24)

time_embedding = tf.feature_column.embedding_column(
    categorical_column=time_,
    dimension=16)

dt_embedding = tf.feature_column.embedding_column(
    categorical_column=dt_,
    dimension=224)

real_fc = [lat, lng, altitude,
           lng_buck_embedding, lat_buck_embedding,
           crossed_ll_embedding, date_embedding, time_embedding,
           dt_embedding, crossed_all_embedding]

all_fc =  [lat, lng, altitude, date_, time_, dt_, lat_buck,
           lng_buck, lng_buck_embedding, lat_buck_embedding,
           crossed_ll_embedding, date_embedding, time_embedding, dt_embedding,
           crossed_all, crossed_all_embedding]


# define all class labels
class_labels = ['bike', 'bus', 'car', 'driving meet conjestion', 'plane', 'subway', 'taxi', 'train', 'walk']
                     
def train_eval(traindir, evaldir, batchsize, bucket, epochs, outputdir, **kwargs):
    # define classifier 
    classifier_config=tf.estimator.RunConfig(save_checkpoints_steps=10, keep_checkpoint_max=500)

    classifier = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=all_fc,
        dnn_feature_columns=real_fc,
        dnn_hidden_units = [15,10,len(class_labels)],
        n_classes=len(class_labels),
        label_vocabulary=class_labels,
        model_dir=outputdir,
        config=classifier_config
    )

    
    # load training and eval files    
    traindata =   [file for file in file_io.get_matching_files(traindir)]
    evaldata =    [file for file in file_io.get_matching_files(evaldir)]

    # define training and eval params
    train_input = lambda: my_input_fn(
            traindata,
            batch_size=batchsize,
            repeat_count = epochs
        )

    eval_input = lambda: my_input_fn(
        evaldata,
        batch_size=1,
        perform_shuffle=False,
        repeat_count = 1
    )

    # define training, eval spec for train and evaluate including
    # exporter for predictions
    train_spec = tf.estimator.TrainSpec(train_input, max_steps=10000)
    exporter = tf.estimator.LatestExporter('exporter',serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                    exporters=[exporter],
                                    name='trajectory-eval',
                                    steps=10,
                                    )                                  
    # run training and evaluation
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)