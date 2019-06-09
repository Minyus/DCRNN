from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def generate_graph_seq2seq_io_data(arr2d,
                                   seq_len,
                                   horizon,
                                   time_in_day_arr1d,
                                   day_of_week_arr1d,
                                   add_time_in_day,
                                   add_day_in_week,
                                   test_timesteps,
                                   val_timesteps
                                   ):

    x_offsets = np.arange(-seq_len + 1, 1, 1)
    y_offsets = np.arange(1, horizon + 1, 1)

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
                     seq_len,
                     horizon,
                     test_samples,
                     val_samples,
                     train_batch_size,
                     test_batch_size,
                     scale,
                     ):

    assert test_samples >= 1
    assert val_samples >= 0

    timesteps_per_sample = (seq_len + horizon)

    test_timesteps = test_samples - 1 + timesteps_per_sample
    val_timesteps = val_samples - 1 + timesteps_per_sample if val_samples > 0 else 0

    assert test_samples >= test_batch_size, \
        'test_samples: {} | test_batch_size:{}'.format(test_samples, test_batch_size)
    assert val_samples == 0 or val_samples >= test_batch_size, \
        'val_samples: {} | test_batch_size:{}'.format(val_samples, test_batch_size)

    if val_timesteps == 0:
        print('Test dataset will be used as validation dataset as well. '
              'To use separate validation dataset, increase val_timesteps. ')

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
                                                          seq_len,
                                                          horizon,
                                                          shuffle=False)
    assert dataloaders['test_loader'].num_batch > 0, 'num_batch for test dataset should be > 0'

    dataloaders['val_loader'] = SpatioTemporalDataLoader(val_z_arr3d, test_batch_size,
                                                         seq_len,
                                                         horizon,
                                                         shuffle=False)
    dataloaders['train_loader'] = SpatioTemporalDataLoader(train_z_arr3d, train_batch_size,
                                                           seq_len,
                                                           horizon,
                                                           shuffle=True)

    dataloaders['scaler'] = scaler
    print('train # batches:', dataloaders['train_loader'].num_batch)
    print('val # batches:', dataloaders['val_loader'].num_batch)
    print('test # batches:', dataloaders['test_loader'].num_batch)

    return dataloaders


class SpatioTemporalDataLoader(object):
    def __init__(self, arr3d, batch_size,
                 seq_len,
                 horizon,
                 shuffle=False,
                 pad_with_last_sample=False):
        """

        :param arr3d:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = max((arr3d.shape[0] - (seq_len + horizon) + 1), 0)
        remainder = (self.size % batch_size)
        if pad_with_last_sample:
            num_padding = (batch_size - remainder)
            x_padding = np.repeat(arr3d[-1:], num_padding, axis=0)
            arr3d = np.concatenate([arr3d, x_padding], axis=0)
            self.size = arr3d.shape[0] - (seq_len + horizon)
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
                    hist_i = i + self.seq_len
                    future_i = hist_i + self.horizon
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

def get_datetime_latest(args):
    if args.latest_timepoint['day_hour_min_option']['set_day_hour_min']:
        d = args.latest_timepoint['day_hour_min_option']['day']
        h = args.latest_timepoint['day_hour_min_option']['hour']
        m = args.latest_timepoint['day_hour_min_option']['min']
        datetime_latest = args.datetime_start + \
                          pd.Timedelta(d - 1, unit='d') + pd.Timedelta(h, unit='h') + pd.Timedelta(m, unit='m')
        print('The latest timepoint ("T") was set to day {:02d} {:02d}:{:02d}'.format(d, h, m))
    elif args.latest_timepoint['timestamp_option']['set_timestamp']:
        ts = args.latest_timepoint['timestamp_option']['timestamp']
        datetime_latest = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        print('The latest timepoint ("T") was set to {}'.format(ts))
    else:
        datetime_latest = df.index.values[-1]
        print('The latest datetime (timestamp "T"): ', datetime_latest)
    return datetime_latest

def generate_train_val_test(args):
    print("Preprocessing data...")

    timestep_size_freq = '{}min'.format(args.timestep_size_in_min)
    timestep_size = pd.Timedelta(args.timestep_size_in_min, unit='m')
    traffic_df_path = Path(args.paths['traffic_df_filename'])
    if traffic_df_path.suffix in ['.h5', '.hdf5']:
        df = pd.read_hdf(args.paths['traffic_df_filename'])
        df.index.name = 'timestamp'
        if not traffic_df_path.with_suffix('.csv').exists():
            df.to_csv(traffic_df_path.with_suffix('.csv').__str__(), sep=',')
    else:
        sep = ',' if traffic_df_path.suffix in ['.csv'] else ' '
        if args.timestep_size_in_min > 0:
            df = pd.read_csv(args.paths['traffic_df_filename'], index_col=False, sep=sep)
            df['timestamp'] = \
                pd.date_range(start='1970-01-01', periods=df.shape[0], freq=timestep_size_freq)
            df = df.set_index('timestamp')
        else:
            df = pd.read_csv(args.paths['traffic_df_filename'], index_col=0, parse_dates=[0], sep=sep)

    args.datetime_start = df.index.values[0]
    args.datetime_latest = get_datetime_latest(args)
    args.datetime_future_start = args.datetime_latest + timestep_size
    args.datetime_future_end = args.datetime_latest + args.model['horizon'] * timestep_size

    d_df = pd.DataFrame()
    d_df['timestamp'] = pd.date_range(start=args.datetime_start, end=args.datetime_future_end,
                                      freq=timestep_size_freq) # Note: end is inclusive.
    d_df = d_df.set_index('timestamp')
    df = pd.merge(d_df, df, how='left', left_index=True, right_index=True)

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
    if args.data['add_time_in_day']:
        arr2d_list.append(broadcast_last_dim(time_in_day_arr1d, arr2d.shape[-1]))
    if args.data['add_day_of_week']:
        arr2d_list.append(broadcast_last_dim(day_of_week_arr1d, arr2d.shape[-1]))

    arr3d = np.stack(arr2d_list, axis=-1)
    num_samples, num_nodes, input_dim = arr3d.shape
    args.model['num_nodes'] = num_nodes
    args.model['input_dim'] = input_dim
    args.model['output_dim'] = 1

    STDATALOADER = True
    if STDATALOADER:
        dataloaders = setup_dataloader(
            arr3d,
            seq_len=args.model['seq_len'],
            horizon=args.model['horizon'],
            test_samples=args.data['test_samples'],
            val_samples=args.data['val_samples'],
            train_batch_size=args.data['train_batch_size'],
            test_batch_size=args.data['test_batch_size'],
            scale=args.data['scale'],
            )
        return dataloaders, args

    if not STDATALOADER:
        x_train, y_train, x_val, y_val, x_test, y_test, x_offsets, y_offsets = \
            generate_graph_seq2seq_io_data(
                arr2d,
                seq_len=args.model['seq_len'],
                horizon=args.model['horizon'],
                time_in_day_arr1d=time_in_day_arr1d,
                day_of_week_arr1d=day_of_week_arr1d,
                add_time_in_day=args.data['add_time_in_day'],
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
        return None, args

def get_adj_mat(args):
    adj_mat_filename = args.paths['adj_mat_filename']
    if Path(adj_mat_filename).suffix in ['.pkl']:
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(adj_mat_filename)
    elif Path(adj_mat_filename).suffix in ['.csv']:
        adj_mx = np.loadtxt(adj_mat_filename, dtype=np.float32, delimiter=',')
    else:
        adj_mx = np.loadtxt(adj_mat_filename, dtype=np.float32, delimiter=' ')
    return adj_mx


def train_dcrnn(args, dataloaders, adj_mx):
    tf.reset_default_graph()
    # with open(args.config_filename) as f:
    #     supervisor_config = yaml.load(f)
    #     args['train']['log_dir'] = args.paths['model_dir']
    #     args['data']['adj_mat_filename'] = args.paths['adj_mat_filename']
    if 1:
        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, dataloaders=dataloaders,
                                         **args)

            supervisor.train(sess=sess)


def get_model_filename(dir):
    path_list = list(Path(dir).glob('*.index'))
    serial = max([int(p.stem.split('-')[-1]) for p in path_list]).__str__()
    path = list(Path(dir).glob('*' + serial + '.index'))[0]
    model_filename = (path.parent / path.stem).as_posix()
    return model_filename


def run_dcrnn(args, dataloaders, adj_mx):

    model_filename = get_model_filename(args.paths['model_dir'])

    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, dataloaders=dataloaders, **args)
        # supervisor.load(sess, config['train']['model_filename'])
        supervisor.load(sess, model_filename)
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.paths['output_filename'], **outputs)
        print('Predictions saved as {}.'.format(args.paths['output_filename']))

    pred = np.load(args.paths['output_filename'], allow_pickle=True)
    pred_tensor = pred['predictions']
    pred_arr2d = pred_tensor[:, -1, :]

    np.savetxt(args.paths['pred_arr2d_file'], pred_arr2d, delimiter=',')


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def read_yaml(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = DotDict({})
    args.update(config)
    return args


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    args = read_yaml('dcrnn_config.yaml')
    dataloaders, args = generate_train_val_test(args)
    adj_mx = get_adj_mat(args)
    if not args.test_only:
        train_dcrnn(args, dataloaders, adj_mx)
    run_dcrnn(args, dataloaders, adj_mx)











