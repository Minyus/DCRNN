from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import geohash
from geopy.distance import great_circle

from lib import utils
from lib.utils import load_graph_data
from lib.array_utils import get_seq_len, ex_partitioned_reduce_mean, broadcast_last_dim

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


def get_geo_df(s_df):
    geo_df = s_df[['geohash6']].drop_duplicates()
    geo_df = geo_df.sort_values(by=['geohash6'])
    geo_df.reset_index(inplace=True, drop=True)
    return geo_df


def get_dist_df(geo_df):

    geo_df.loc[:, 'lat'] = geo_df['geohash6'].apply(lambda x: geohash.decode(x)[0])
    geo_df.loc[:, 'lng'] = geo_df['geohash6'].apply(lambda x: geohash.decode(x)[1])

    geo_df['cost'] = 0.0

    geo2_df = pd.merge(geo_df, geo_df, on='cost', how='outer')

    geo2_df.loc[:,'cost'] = geo2_df[['lat_x', 'lng_x', 'lat_y', 'lng_y']].\
        apply(lambda x: great_circle((x[0], x[1]), (x[2], x[3])).km, axis=1)

    dist_df = geo2_df.loc[:,['geohash6_x', 'geohash6_y', 'cost']].rename(columns={'geohash6_x':'from', 'geohash6_y':'to'})

    return dist_df


def get_spatiotemporal_df(s_df, args, node_ids=None):

    def get_datetime(x):
        dt = datetime(1970, 1, 1) + \
            pd.Timedelta(x[0] - 1, unit='d') + \
            pd.Timedelta(int(x[1].split(':')[0]), unit='h') + \
            pd.Timedelta(int(x[1].split(':')[1]), unit='m')
        return dt

    s_df['datetime'] = \
        s_df[['day', 'timestamp']].apply(get_datetime, axis=1)

    wide_df = s_df.pivot(index='datetime', columns='geohash6', values='demand')
    if node_ids is not None:
        # Make sure the column order is as in the adjacency matrix
        wide_df = wide_df[node_ids]

    dt_df = pd.DataFrame()
    dt_df.loc[:, 'datetime'] = pd.date_range(start=s_df['datetime'].min(), end=s_df['datetime'].max(),
                  freq=args.timestep_size_freq)
    dt_df.set_index('datetime',inplace=True)

    st_df = pd.merge(dt_df, wide_df, how='left', left_index=True, right_index=True)

    st_df.fillna(0.0, inplace=True)

    st_df.index.name = 'timestamp'

    return st_df


def get_adjacency_matrix(distance_df, node_ids, normalized_k=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param node_ids: list of node ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_nodes = len(node_ids)
    dist_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    node_id_to_ind = {}
    for i, node_id in enumerate(node_ids):
        node_id_to_ind[node_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in node_id_to_ind or row[1] not in node_id_to_ind:
            continue
        dist_mx[node_id_to_ind[row[0]], node_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return node_ids, node_id_to_ind, adj_mx


def get_adj_mat(args, adj_mx=None):
    if adj_mx is None:
        adj_mat_filename = args.paths['adj_mat_filename']
        if Path(adj_mat_filename).suffix in ['.pkl']:
            sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(adj_mat_filename)
        elif Path(adj_mat_filename).suffix in ['.csv']:
            adj_mx = np.loadtxt(adj_mat_filename, dtype=np.float32, delimiter=',')
        else:
            adj_mx = np.loadtxt(adj_mat_filename, dtype=np.float32, delimiter=' ')
    return adj_mx


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

    x = [data[t + x_offsets, ...] for t in range(min_t, max_t)]
    y = [data[t + y_offsets, ...] for t in range(min_t, max_t)]

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)

    print("history (model_input): ", x.shape, " | future (model_output): ", y.shape)
    # Write the data into npz file.
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
                     val_batch_size,
                     test_batch_size,
                     scale,
                     time_dim_index=None
                     ):

    assert test_samples >= 1
    assert val_samples >= 0

    timesteps_per_sample = (sum(seq_len) if isinstance(seq_len, list) else seq_len) + horizon

    test_timesteps = test_samples - 1 + timesteps_per_sample
    val_timesteps = val_samples - 1 + timesteps_per_sample if val_samples > 0 else 0

    assert test_samples >= test_batch_size, \
        'test_samples: {} | test_batch_size:{}'.format(test_samples, test_batch_size)
    assert val_samples == 0 or val_samples >= val_batch_size, \
        'val_samples: {} | val_batch_size:{}'.format(val_samples, val_batch_size)

    if val_timesteps == 0:
        print('Test dataset will be used as validation dataset as well. '
              'To use separate validation dataset, increase val_timesteps. ')

    num_samples, num_nodes, _ = arr3d.shape

    num_test = test_timesteps if test_timesteps <= num_samples else num_samples
    num_val = val_timesteps if \
        (test_timesteps + val_timesteps) <= num_samples else 0
    num_train = num_samples - num_test - num_val

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
    dataloaders['test_loader'] = \
        SpatioTemporalDataLoader(test_z_arr3d, test_batch_size, seq_len, horizon, shuffle=False,
                                 time_dim_index=time_dim_index)
    assert dataloaders['test_loader'].num_batch > 0, 'num_batch for test dataset should be > 0'
    dataloaders['val_loader'] = \
        SpatioTemporalDataLoader(val_z_arr3d, val_batch_size, seq_len, horizon, shuffle=False,
                                 time_dim_index=time_dim_index)
    dataloaders['train_loader'] = \
        SpatioTemporalDataLoader(train_z_arr3d, train_batch_size, seq_len, horizon, shuffle=True,
                                 time_dim_index=time_dim_index)

    dataloaders['scaler'] = scaler
    print('[train]      | # timesteps: {:06d} | # samples: {:06d} | # batches: {:06d}'.\
          format(num_train, dataloaders['train_loader'].size, dataloaders['train_loader'].num_batch))
    print('[validation] | # timesteps: {:06d} | # samples: {:06d} | # batches: {:06d}'.\
          format(num_val, dataloaders['val_loader'].size, dataloaders['val_loader'].num_batch))
    print('[test]       | # timesteps: {:06d} | # samples: {:06d} | # batches: {:06d}'.\
          format(num_test, dataloaders['test_loader'].size, dataloaders['test_loader'].num_batch))

    return dataloaders


class SpatioTemporalDataLoader(object):
    def __init__(self, arr3d, batch_size,
                 seq_len,
                 horizon,
                 shuffle=False,
                 pad_with_last_sample=False,
                 time_dim_index=None):

        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.time_dim_index = time_dim_index
        self.current_ind = 0
        self.size = max((arr3d.shape[0] - \
                         ((sum(seq_len) if isinstance(seq_len, list) else seq_len) + horizon) + 1), 0)
        remainder = (self.size % batch_size)
        # pad with the last sample to make number of samples divisible to batch_size.
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

        enable_seq_reducing = isinstance(self.seq_len, list)
        seq_len = sum(self.seq_len) if enable_seq_reducing else self.seq_len

        def _wrapper():
            while self.current_ind < self.num_batch:

                x_arr3d_list, y_arr3d_list = [], []
                for _ in range(self.batch_size):
                    i = sample_index_list.pop()

                    hist_i = i + seq_len
                    x_arr3d = ex_partitioned_reduce_mean(self.arr3d, i, part_size_list=self.seq_len) \
                        if enable_seq_reducing \
                        else self.arr3d[i:hist_i]
                    if self.time_dim_index is not None:
                        # extract the fractional part (time) to remove the integer part (day)
                        x_arr3d[..., self.time_dim_index] = x_arr3d[..., self.time_dim_index] % 1
                    x_arr3d_list.append(x_arr3d)

                    future_i = hist_i + self.horizon
                    y_arr3d_list.append(self.arr3d[hist_i:future_i])
                x_arr4d = np.stack(x_arr3d_list)
                y_arr4d = np.stack(y_arr3d_list)
                yield (x_arr4d, y_arr4d)
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


def get_datetime_latest(args, df):
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


def generate_train_val_test(args, df=None):

    if df is None:
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
                    pd.date_range(start='1970-01-01', periods=df.shape[0], freq=args.timestep_size_freq)
                df = df.set_index('timestamp')
            else:
                df = pd.read_csv(args.paths['traffic_df_filename'], index_col=0, parse_dates=[0], sep=sep)

    args.datetime_start = df.index.values[0]
    args.datetime_latest = get_datetime_latest(args, df)
    timestep_size = pd.Timedelta(args.timestep_size_in_min, unit='m')
    args.datetime_future_start = args.datetime_latest + timestep_size
    args.datetime_future_end = args.datetime_latest + args.model['horizon'] * timestep_size

    d_df = pd.DataFrame()
    d_df['timestamp'] = pd.date_range(start=args.datetime_start, end=args.datetime_future_end,
                                      freq=args.timestep_size_freq) # Note: end is inclusive.
    d_df = d_df.set_index('timestamp')
    df = pd.merge(d_df, df, how='left', left_index=True, right_index=True)

    arr2d = df.values
    arr2d_list = [arr2d]

    # _day_arr1d = df.index.values.astype("datetime64[D]")
    # time_in_day_arr1d = (df.index.values - _day_arr1d) / np.timedelta64(1, "D")

    # Note: extract the fractional part (time) after reduce_mean
    time_in_day_arr1d = (df.index - datetime(1970, 1, 1)).total_seconds() / \
                        timedelta(days=1).total_seconds()
    day_of_week_arr1d = df.index.dayofweek

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
        args.model['train_steps_per_epoch'] = \
            args.data['train_samples_per_epoch'] // args.data['train_batch_size']
        args.model['target_train_steps'] = \
            args.data['target_train_samples'] // args.data['train_batch_size']

        if args.model.get('seq_reducing') and \
                args.model.get('seq_reducing').get('enable_seq_reducing'):
            seq_len = args.model.get('seq_reducing').get('seq_len_list')
            args.model['seq_len'] = get_seq_len(seq_len)
        else:
            seq_len = args.model['seq_len']
        dataloaders = setup_dataloader(
            arr3d,
            seq_len=seq_len,
            horizon=args.model['horizon'],
            test_samples=args.data['test_samples_per_epoch'],
            val_samples=args.data['val_samples_per_epoch'],
            train_batch_size=args.data['train_batch_size'],
            val_batch_size=args.data['val_batch_size'],
            test_batch_size=args.data['test_batch_size'],
            scale=args.data['scale'],
            time_dim_index=1 if args.data['add_time_in_day'] else None,
            )
        return args, dataloaders

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
        return args, None


def preprocess(args):
    logger = utils.get_logger(args.paths['model_dir'], __name__, level=args.get('log_level', 'INFO'))
    logger.info('Started preprocessing...')

    args.timestep_size_freq = '{}min'.format(args.timestep_size_in_min)

    need_st_flag = not Path(args.paths['traffic_df_filename']).exists()
    need_adj_flag = (not Path(args.paths['adj_mat_filename']).exists()) or \
                    (not Path(args.paths['geohash6_filename']).exists())

    if need_st_flag or need_adj_flag:

        source_table_dir = Path(args.paths.get('source_table_dir'))
        source_table_filename = source_table_dir.glob('*.csv').__next__()
        if not source_table_filename:
            raise FileNotFoundError('directory: ' + args.paths.get('source_table_dir'))
        logger.info('Reading: {}'.format(source_table_filename))
        s_df = pd.read_csv(source_table_filename)

    adj_mx = None
    node_ids = None
    if need_adj_flag:
        logger.info('Preparing adjacency matrix...')
        ##%% Prepare for adjacency matrix
        geo_df = get_geo_df(s_df)
        node_ids = geo_df['geohash6'].values
        with open(args.paths['geohash6_filename'], 'w') as f:
            f.write(','.join(node_ids))
        dist_df = get_dist_df(geo_df)
        dist_df.to_csv(args.paths.get('distances_filename'), index=False)

        # with open(args.paths.get('geohash6_filename')) as f:
        #     node_ids = f.read().strip().split(',')

        # distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
        distance_df = dist_df

        _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, node_ids)

        # Save to pickle file.
        # with open(args.output_pkl_filename, 'wb') as f:
        #     pickle.dump([node_ids, sensor_id_to_ind, adj_mx], f, protocol=2)

        np.savetxt(args.paths['adj_mat_filename'], adj_mx, delimiter=',')
        logger.info('Adjacency matrix was saved at: {}'.format(args.paths['adj_mat_filename']))

    else:
        node_ids_df = pd.read_csv(args.paths.get('geohash6_filename'), index_col=None, sep=',')
        node_ids = node_ids_df.columns
        adj_mx = get_adj_mat(args, adj_mx)

    st_df = None
    if need_st_flag:
        logger.info('Preparing spatio-temporal dataframe...')
        # Get Spatio-temporal df
        st_df = get_spatiotemporal_df(s_df, args, node_ids)
        st_df.to_csv(args.paths['traffic_df_filename'])
        logger.info('Spatio-temporal dataframe was saved at: {}'.format(args.paths['traffic_df_filename']))

    args, dataloaders = generate_train_val_test(args, st_df)

    logger.info('Completed preprocessing.')

    return args, dataloaders, adj_mx, node_ids


def transform_to_long(pred_df=None):
    # if pred_df is None:
    #     pred_df = pd.read_csv(args.paths['pred_df_filename'], parse_dates=[0], index_col=0)
    long_df = pd.DataFrame(pred_df.stack(), columns=['demand'])
    long_df.reset_index(inplace=True)
    long_df.loc[:, 'day'] = long_df['timestamp'].apply(lambda x: int((x - datetime(1970, 1, 1)).days) + 1)
    long_df.loc[:, 'timestamp'] = long_df['timestamp'].apply(lambda x: x.strftime('%H:%M'))
    long_df.rename(columns={'level_1': 'geohash6'}, inplace=True)
    long_df = long_df[['geohash6', 'day', 'timestamp', 'demand']]
    return long_df


def save_pred_long_df(args, long_df):
    logger = utils.get_logger(args.paths['model_dir'], __name__, level=args.get('log_level', 'INFO'))
    long_df.to_csv(args.paths['pred_long_filename'], index=False)
    logger.info('The final prediction output file was saved at: {}'.\
          format(args.paths['pred_long_filename']))