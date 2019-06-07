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
import random

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor




import argparse
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def generate_graph_seq2seq_io_data(arr2d,
                                   history_timesteps,
                                   future_timesteps,
                                   time_in_day_arr1d,
                                   day_of_week_arr1d,
                                   add_time_in_day,
                                   add_day_in_week,
                                   test_timesteps,
                                   val_timesteps
                                   ):

    x_offsets = np.arange(-history_timesteps + 1, 1, 1)
    y_offsets = np.arange(1, future_timesteps + 1, 1)

    num_samples, num_nodes = arr2d.shape
    data = np.expand_dims(arr2d, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_in_day = np.tile(time_in_day_arr1d, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, day_of_week_arr1d] = 1
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

    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)

    print("history (model_input): ", x.shape, " | future (model_output): ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]

    num_test = test_timesteps if test_timesteps <= num_samples else num_samples
    num_val = val_timesteps if \
        (test_timesteps + val_timesteps) <= num_samples else 0
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

    return x_train, y_train, x_val, y_val, x_test, y_test, x_offsets, y_offsets


def setup_dataloader(arr3d,
                     history_timesteps,
                     future_timesteps,
                     test_timesteps,
                     val_timesteps,
                     train_batch_size,
                     test_batch_size,
                     scale,
                    ):
    min_timesteps = (history_timesteps + future_timesteps)
    if ((test_timesteps - min_timesteps + 1) < test_batch_size):
        test_timesteps = test_batch_size + min_timesteps - 1
        print('test_timesteps:', test_timesteps)
        print('test_batch_size', test_batch_size)
        print('test_timesteps was set to too small. Set to the minimum value:', test_timesteps)
    if ((val_timesteps - min_timesteps + 1) < test_batch_size):
        print('val_timesteps:', val_timesteps)
        print('test_batch_size', test_batch_size)
        print('Test dataset will be used as validation dataset as well. '
              'To use separate validation dataset, increase val_timesteps. ')
        val_timesteps = 0
        

    num_samples, num_nodes, _ = arr3d.shape

    num_test = test_timesteps if test_timesteps <= num_samples else num_samples
    num_val = val_timesteps if \
        (test_timesteps + val_timesteps) <= num_samples else 0
    num_train = num_samples - num_test - num_val
    print(' num_train: {} \n num_val: {} \n num_test: {}'.format(num_train, num_val, num_test))

    test_arr3d = arr3d[-num_test:]
    val_arr3d = arr3d[num_train: num_train + num_val] if num_val > 0 else test_arr3d
    train_arr3d = arr3d[:num_train]

    train_arr2d = train_arr3d[:, :, 0]
    val_arr2d = val_arr3d[:, :, 0]
    test_arr2d = test_arr3d[:, :, 0]

    train_z_arr3d = train_arr3d.copy()
    val_z_arr3d = val_arr3d.copy()
    test_z_arr3d = test_arr3d.copy()

    scaler = StandardScaler(mean=train_arr2d.mean(), std=train_arr2d.std(),
                            scale=scale)
    train_z_arr3d[:, :, 0] = scaler.transform(train_arr2d)
    val_z_arr3d[:, :, 0] = scaler.transform(val_arr2d)
    test_z_arr3d[:, :, 0] = scaler.transform(test_arr2d)

    dataloaders = {}
    dataloaders['test_loader'] = SpatioTemporalDataLoader(test_z_arr3d, test_batch_size,
                                                   history_timesteps,
                                                   future_timesteps,
                                                   shuffle=False)
    assert dataloaders['test_loader'].num_batch > 0, 'num_batch for test dataset should be > 0'

    dataloaders['val_loader'] = SpatioTemporalDataLoader(val_z_arr3d, test_batch_size,
                                                  history_timesteps,
                                                  future_timesteps,
                                                  shuffle=False)
    dataloaders['train_loader'] = SpatioTemporalDataLoader(train_z_arr3d, train_batch_size,
                                                    history_timesteps,
                                                    future_timesteps,
                                                    shuffle=True)

    dataloaders['scaler'] = scaler
    print('train # batches:', dataloaders['train_loader'].num_batch)
    print('val # batches:', dataloaders['val_loader'].num_batch)
    print('test # batches:', dataloaders['test_loader'].num_batch)

    return dataloaders


class SpatioTemporalDataLoader(object):
    def __init__(self, arr3d, batch_size,
                 history_timesteps,
                 future_timesteps,
                 shuffle=False,
                 pad_with_last_sample=False):
        """

        :param arr3d:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.history_timesteps = history_timesteps
        self.future_timesteps = future_timesteps
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = max((arr3d.shape[0] - (history_timesteps + future_timesteps) + 1), 0)
        remainder = (self.size % batch_size)
        if pad_with_last_sample:
            num_padding = (batch_size - remainder)
            x_padding = np.repeat(arr3d[-1:], num_padding, axis=0)
            arr3d = np.concatenate([arr3d, x_padding], axis=0)
            self.size = arr3d.shape[0] - (history_timesteps + future_timesteps)
        else:
            # drop first
            arr3d = arr3d[remainder:]

        self.num_batch = int(self.size // self.batch_size)

        self.size = self.num_batch * self.batch_size

        # if shuffle:
        #     permutation = np.random.permutation(self.size)
        #     xs = xs[permutation]
        self.shuffle = shuffle
        self.arr3d = arr3d


    def get_iterator(self):
        self.current_ind = 0

        def range_list(size, shuffle=False):
            seq = range(size).__reversed__()
            out = list(seq)
            if shuffle:
                random.shuffle(out)
            return out

        sample_index_list = range_list(self.size, shuffle=self.shuffle)

        def _wrapper():
            while self.current_ind < self.num_batch:

                x_i, y_i = [], []
                for _ in range(self.batch_size):
                    i = sample_index_list.pop()
                    hist_i = i + self.history_timesteps
                    future_i = hist_i + self.future_timesteps
                    x_i.append(self.arr3d[i:hist_i])
                    y_i.append(self.arr3d[hist_i:future_i])
                x_i = np.stack(x_i)
                y_i = np.stack(y_i)
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std, scale=True):
        self.mean = mean
        self.std = std
        self.scale = scale

    def transform(self, data):
        return (data - self.mean) / self.std if self.scale else data

    def inverse_transform(self, data):
        return (data * self.std) + self.mean if self.scale else data





def generate_train_val_test(args):
    print("Converting data from 2d of (epoch_timesteps, num_nodes) \n"
          "to 4d of (epoch_timesteps, model_timesteps, num_nodes, dimension).")

    assert args.test_timesteps >= 0
    assert args.val_timesteps >= 0

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

    arr2d = df.values
    time_in_day_arr1d = (df.index.values - df.index.values.astype("datetime64[D]")) / \
                        np.timedelta64(1, "D")
    day_of_week_arr1d = df.index.dayofweek

    def broadcast_last_dim(arr1d, num_broadcast):
        arr2d = np.expand_dims(arr1d, -1)
        arr2d = np.tile(arr2d, (1, num_broadcast))
        return arr2d

    broadcast_last_dim(time_in_day_arr1d, arr2d.shape[-1])

    arr2d_list = [arr2d]
    if args.add_time_in_day:
        arr2d_list.append(broadcast_last_dim(time_in_day_arr1d, arr2d.shape[-1]))
    if args.add_day_of_week:
        arr2d_list.append(broadcast_last_dim(day_of_week_arr1d, arr2d.shape[-1]))

    arr3d = np.stack(arr2d_list, axis=-1)

    STDATALOADER = True
    if STDATALOADER:
        dataloaders = setup_dataloader(
            arr3d,
            history_timesteps=args.history_timesteps,
            future_timesteps=args.future_timesteps,
            test_timesteps=args.test_timesteps,
            val_timesteps=args.val_timesteps,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            scale=args.scale,
            )
        return dataloaders

    if not STDATALOADER:
        x_train, y_train, x_val, y_val, x_test, y_test, x_offsets, y_offsets = \
            generate_graph_seq2seq_io_data(
                arr2d,
                history_timesteps=args.history_timesteps,
                future_timesteps=args.future_timesteps,
                time_in_day_arr1d=time_in_day_arr1d,
                day_of_week_arr1d=day_of_week_arr1d,
                add_time_in_day=args.add_time_in_day,
                add_day_in_week=args.add_day_in_week,
                test_timesteps=args.test_timesteps,
                val_timesteps=args.val_timesteps,
                )

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

def train_dcrnn(args, dataloaders):
    tf.reset_default_graph()
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['train']['log_dir'] = args.model_dir
        supervisor_config['data']['graph_pkl_filename'] = args.graph_pkl_filename
        # graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        graph_pkl_filename = args.graph_pkl_filename
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, dataloaders=dataloaders, **supervisor_config)

            supervisor.train(sess=sess)

def get_model_filename(dir):
    path_list = list(Path(dir).glob('*.index'))
    serial = max([int(p.stem.split('-')[-1]) for p in path_list]).__str__()
    path = list(Path(dir).glob('*' + serial + '.index'))[0]
    model_filename = (path.parent / path.stem).as_posix()
    return model_filename


def run_dcrnn(args, dataloaders):
    tf.reset_default_graph()
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    # graph_pkl_filename = config['data']['graph_pkl_filename']
    graph_pkl_filename = args.graph_pkl_filename
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, dataloaders=dataloaders, **config)
        # supervisor.load(sess, config['train']['model_filename'])
        model_filename = get_model_filename(args.model_dir)
        supervisor.load(sess, model_filename)
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))

    pred = np.load(args.output_filename, allow_pickle=True)
    pred_tensor = pred['predictions']
    pred_arr2d = pred_tensor[:, -1, :]

    np.savetxt(args.pred_arr2d_file, pred_arr2d, delimiter=',')

def main(args):
    dataloaders = generate_train_val_test(args)
    if not args.test_only:
        train_dcrnn(args, dataloaders)
    run_dcrnn(args, dataloaders)

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", type=bool, default=True,
                        help="Skip training",)

    parser.add_argument("--traffic_df_filename", type=str, default="metr-la_data/metr-la.csv",
                        help="Raw traffic readings.",)
    parser.add_argument("--graph_pkl_filename", type=str, default="metr-la_data/adj_mx.pkl",
                        help="Graph adjacency matrix.",)
    parser.add_argument("--model_dir", type=str, default="metr-la_data/model",
                        help="model directory.",)

    parser.add_argument("--output_dir", type=str, default="data/METR-LA",
                        help="Output directory.",)

    parser.add_argument("--history_timesteps", type=int, default=12,
                        help="timesteps to use as model input.",)
    parser.add_argument("--future_timesteps", type=int, default=12,
                        help="timesteps to predict by the model.",)

    parser.add_argument("--test_timesteps", type=int, default=88,
                        help="timesteps for test.",)
    parser.add_argument("--val_timesteps", type=int, default=0,
                        help="timesteps for validation. "
                             "if 0, test dataset is used as validation.",)
    parser.add_argument("--add_time_in_day", type=bool, default=True,
                        help="Add time in day to the model input dimensions.",)
    parser.add_argument("--add_day_of_week", type=bool, default=False,
                        help="Add day of week to the model input dimensions.",)
    parser.add_argument("--timestep_size_in_min", type=int, default=5,
                        help="Specify the timestep size in minutes.",)
    parser.add_argument("--timestamp_latest", type=str, default=None,
                        help="The timestamp of the latest datetime in "
                             "%Y-%m-%dT%H:%M:%S format e.g. '1970-02-15T18:00:00' ",)
    parser.add_argument("--day_hour_min_latest", type=str, default='1_18_55',
                        help="day, hour, minute of the latest datetime in "
                             " dd_hh_mm format e.g. 50_18_15 ",)

    # parser.add_argument('--config_filename', default=None, type=str,
    #                     help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')


    # parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='metr-la_data/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')

    parser.add_argument('--pred_arr2d_file', default='data/dcrnn_pred_arr2d.csv')

    parser.add_argument('--train_batch_size', type=int, default=64)
    # parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help="batch size for test and validation.")

    parser.add_argument('--scale', type=bool, default=False)

    args = parser.parse_args()
    main(args)











