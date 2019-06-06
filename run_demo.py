from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from datetime import datetime

import numpy as np
import os
import pandas as pd
from pathlib import Path


import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor



def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False,
        ):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) /\
                   np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)

    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive

    # x, y = [], []
    # for t in range(min_t, max_t):
    #     x_t = data[t + x_offsets, ...]
    #     y_t = data[t + y_offsets, ...]
    #     x.append(x_t)
    #     y.append(y_t)
    x = [data[t + x_offsets, ...] for t in range(min_t, max_t)]
    y = [data[t + y_offsets, ...] for t in range(min_t, max_t)]

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    print("Converting data from 2d of (epoch_timesteps, num_nodes) \n"
          "to 4d of (epoch_timesteps, model_timesteps, num_nodes, dimension).")
    origin = '1970-01-01'
    traffic_df_path = Path(args.traffic_df_filename)
    if traffic_df_path.suffix in ['.h5', '.hdf5']:
        df = pd.read_hdf(args.traffic_df_filename)
        df.index.name = 'timestamp'
        if not traffic_df_path.with_suffix('.csv').exists():
            df.to_csv(traffic_df_path.with_suffix('.csv').__str__(), sep=',')
    else:
        sep = ',' if traffic_df_path.suffix in ['.csv'] else ' '
        if args.timestep_size_in_min > 0:
            freq = '{}min'.format(args.timestep_size_in_min)
            df = pd.read_csv(args.traffic_df_filename, index_col=False, sep=sep)
            df['timestamp'] = pd.date_range(start=origin, periods=df.shape[0], freq=freq)
            df = df.set_index('timestamp')
        else:
            df = pd.read_csv(args.traffic_df_filename, index_col=0, parse_dates=[0], sep=sep)

    timestep_size = pd.Timedelta(args.timestep_size_in_min, unit='m')
    assert (args.timestamp_latest is None) or (args.day_hour_min_latest is None)
    if args.day_hour_min_latest:
        print('Exclude future samples after future_timesteps since day_hour_min_latest is specified as: \n',
              args.day_hour_min_latest)
        d, h, m = [int(e) for e in args.day_hour_min_latest.split('_')]
        args.datetime_latest = datetime.strptime(origin, "%Y-%m-%d") + \
            pd.Timedelta(d-1, unit='d') + pd.Timedelta(h, unit='h') + pd.Timedelta(m, unit='m')
        df = df.loc[:(args.datetime_latest + args.future_timesteps * timestep_size)]  # Note: .loc is inclusive

    if args.timestamp_latest:
        print('Exclude future samples after future_timesteps since timestamp_latest is specified as: \n',
              args.timestamp_latest)
        args.datetime_latest = datetime.strptime(args.timestamp_latest, "%Y-%m-%dT%H:%M:%S")
        df = df.loc[:(args.datetime_latest + args.future_timesteps * timestep_size)] # Note: .loc is inclusive

    # Keep timestamp info to output final prediction table
    args.datetime_max = df.index.values[-1]
    print('The maximum of datetime: ', args.datetime_max)
    args.datetime_latest = args.datetime_max - \
                            args.future_timesteps * timestep_size

    args.datetime_test_init = args.datetime_latest + timestep_size

    x_offsets = np.arange(-args.history_timesteps + 1, 1, 1)
    y_offsets = np.arange(1, args.future_timesteps + 1, 1)

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week,
    )
    print("history (model_input): ", x.shape, " | future (model_output): ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    # num_test = round(num_samples * 0.2)
    # num_train = round(num_samples * 0.7)
    # num_val = num_samples - num_test - num_train
    assert args.test_timesteps >= 0
    assert args.validation_timesteps >= 0
    num_test = args.test_timesteps if args.test_timesteps <= num_samples else num_samples
    num_val = args.validation_timesteps if \
        (args.test_timesteps + args.validation_timesteps) <= num_samples else 0
    num_train = num_samples - num_test - num_val

    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
        ) if num_val > 0 else (x_test, y_test)

    # train
    x_train, y_train = x[:num_train], y[:num_train]



    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, " >> history (model_input):", _x.shape, " | future (model_output): ", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def run_dcrnn(args):
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        supervisor.load(sess, config['train']['model_filename'])
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))

    pred = np.load(args.output_filename, allow_pickle=True)
    pred_tensor = pred['predictions']
    pred_arr2d = pred_tensor[:, -1, :]

    np.savetxt(args.pred_arr2d_file, pred_arr2d, delimiter=',')

def main(args):
    generate_train_val_test(args)
    run_dcrnn(args)

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/METR-LA",
                        help="Output directory.",)
    parser.add_argument("--traffic_df_filename", type=str, default="data/metr-la.csv",
                        help="Raw traffic readings.",)
    parser.add_argument("--history_timesteps", type=int, default=12,
                        help="timesteps to use as model input.",)
    parser.add_argument("--future_timesteps", type=int, default=12,
                        help="timesteps to predict by the model.",)
    # parser.add_argument("--phase", type=str, default='train_validation_test',
    #                     help="train_validation_test, train_validation, or 'test'",)
    parser.add_argument("--test_timesteps", type=int, default=12,
                        help="timesteps for test.",)
    parser.add_argument("--validation_timesteps", type=int, default=0,
                        help="timesteps for validation. "
                             "if 0, test dataset is used as validation.",)
    parser.add_argument("--add_time_in_day", type=bool, default=True,
                        help="Add time in day to the model input dimensions.",)
    parser.add_argument("--add_day_in_week", type=bool, default=False,
                        help="Add day in week to the model input dimensions.",)
    parser.add_argument("--timestep_size_in_min", type=int, default=5,
                        help="Specify the timestep size in minutes.",)
    parser.add_argument("--timestamp_latest", type=str, default=None,
                        help="The timestamp of the latest datetime in "
                             "%Y-%m-%dT%H:%M:%S format e.g. '1970-02-15T18:00:00' ",)
    parser.add_argument("--day_hour_min_latest", type=str, default='1_02_55',
                        help="day, hour, minute of the latest datetime in "
                             " dd_hh_mm format e.g. 50_18_15 ",)

    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')

    parser.add_argument('--pred_arr2d_file', default='data/dcrnn_pred_arr2d.csv')

    args = parser.parse_args()
    main(args)









