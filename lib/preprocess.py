from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat

import pandas as pd
import numpy as np
import geohash
from geopy.distance import great_circle

from lib.utils import load_graph_data, StandardScaler
from lib.array_utils import get_seq_len, ex_partitioned_reduce_mean, broadcast_last_dim
from lib.logging_utils import config_logging


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


def get_adjacency_matrix(distance_df, node_ids, proximity_threshold=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param node_ids: list of node ids.
    :param proximity_threshold: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_nodes = len(node_ids)
    dist_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds node id to index map.
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
    adj_mx[adj_mx < proximity_threshold] = 0
    return node_ids, node_id_to_ind, adj_mx


def get_adj_mat(args, geo_df, node_ids):
    if geo_df is not None:
        dist_df = get_dist_df(geo_df)

        distances_filename = args.paths.get('distances_filename')
        if distances_filename is not None:
            dist_df.to_csv(distances_filename, index=False)

        proximity_threshold = args.get('model').get('proximity_threshold', 0.1)\
            if args.get('model') else 0.1
        _, _, adj_mx = get_adjacency_matrix(dist_df, node_ids, proximity_threshold=proximity_threshold)

        adj_mat_filename = args.paths.get('adj_mat_filename')
        if adj_mat_filename is not None:
            np.savetxt(adj_mat_filename, adj_mx, delimiter=',')
            print('Adjacency matrix was saved at: {}'.format(adj_mat_filename))

    return adj_mx


def setup_dataloader(arr3d,
                     seq_len,
                     horizon,
                     length_dict,
                     train_batch_size,
                     val_batch_size,
                     test_batch_size,
                     scale,
                     features,
                     logger,
                     ):

    train_length = length_dict['train_length']
    val_length = length_dict['val_length']
    test_length = length_dict['test_length']

    test_arr3d = arr3d[-test_length:]
    val_arr3d = arr3d[train_length: train_length + val_length] if val_length > 0 else test_arr3d
    train_arr3d = arr3d[:train_length]

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
                                 features=features)
    assert dataloaders['test_loader'].num_batch > 0, 'num_batch for test dataset should be > 0'

    dataloaders['val_loader'] = \
        SpatioTemporalDataLoader(val_z_arr3d, val_batch_size, seq_len, horizon, shuffle=False,
                                 features=features)
    dataloaders['train_loader'] = \
        SpatioTemporalDataLoader(train_z_arr3d, train_batch_size, seq_len, horizon, shuffle=True,
                                 features=features)

    dataloaders['scaler'] = scaler
    logger.info('[train]      | # timesteps: {:06d} | # samples: {:06d} | # batches: {:06d}'.\
          format(train_length, dataloaders['train_loader'].size, dataloaders['train_loader'].num_batch))
    logger.info('[validation] | # timesteps: {:06d} | # samples: {:06d} | # batches: {:06d}'.\
          format(val_length, dataloaders['val_loader'].size, dataloaders['val_loader'].num_batch))
    logger.info('[test]       | # timesteps: {:06d} | # samples: {:06d} | # batches: {:06d}'.\
          format(test_length, dataloaders['test_loader'].size, dataloaders['test_loader'].num_batch))

    return dataloaders


class SpatioTemporalDataLoader(object):
    def __init__(self, arr3d, batch_size,
                 seq_len,
                 horizon,
                 shuffle=False,
                 pad_with_last_sample=False,
                 features=None):

        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.features = features
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
                    x_arr3d = ex_partitioned_reduce_mean(self.arr3d, i,
                                                         part_size_list=self.seq_len) \
                        if enable_seq_reducing \
                        else self.arr3d[i:hist_i]
                    if self.features.get('add_time_in_day'):
                        # extract the fractional part (time) to remove the integer part (day)
                        x_arr3d[..., 1] = x_arr3d[..., 1] % 1
                    if self.features.get('add_day_of_week'):
                        x_arr3d[..., -1] = np.floor(x_arr3d[..., -1]) % 7

                    x_arr3d_list.append(x_arr3d)

                    future_i = hist_i + self.horizon
                    y_arr3d_list.append(self.arr3d[hist_i:future_i])
                x_arr4d = np.stack(x_arr3d_list)
                y_arr4d = np.stack(y_arr3d_list)
                yield (x_arr4d, y_arr4d)
                self.current_ind += 1

        return _wrapper()


def get_datetime_latest(args, df):
    logger = args.logger
    if args.latest_timepoint['day_hour_min_option']['set_day_hour_min']:
        d = args.latest_timepoint['day_hour_min_option']['day']
        h = args.latest_timepoint['day_hour_min_option']['hour']
        m = args.latest_timepoint['day_hour_min_option']['min']
        datetime_latest = args.datetime_start + \
                          pd.Timedelta(d - 1, unit='d') + pd.Timedelta(h, unit='h') + pd.Timedelta(m, unit='m')
        logger.info('The latest timepoint ("T") is set to day {:02d} {:02d}:{:02d}'.format(d, h, m))
    elif args.latest_timepoint['timestamp_option']['set_timestamp']:
        ts = args.latest_timepoint['timestamp_option']['timestamp']
        datetime_latest = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        logger.info('The latest timepoint ("T") is set to {}'.format(ts))
    else:
        datetime_latest = df.index.values[-1]
        logger.info('The latest timepoint ("T") is: ', datetime_latest)
    return datetime_latest


def aggregate_keeping_rows(df, agg_samples):
    dt_df = df.copy()
    dt_list = ['hour', 'dayofweek']
    dt_df.loc[:, dt_list[0]] = dt_df.index.hour
    dt_df.loc[:, dt_list[1]] = dt_df.index.dayofweek

    agg_df = dt_df[:agg_samples].groupby(dt_list, as_index=False).mean()

    dt_df.index.name = 'timestamp'

    agg_df = pd.merge(dt_df.reset_index().drop(list(df.columns.values), axis=1), agg_df, how='left', on=dt_list)
    agg_df = agg_df.set_index('timestamp')
    agg_df = agg_df.drop(dt_list, axis=1)
    return agg_df


def generate_train_val_test(args, df=None):
    logger = args.logger

    assert args.data['train_samples_per_epoch'] >= args.data['train_batch_size']
    assert args.data['target_train_samples'] >= args.data['train_samples_per_epoch']

    args.data['train_steps_per_epoch'] = \
        args.data['train_samples_per_epoch'] // args.data['train_batch_size']
    args.data['target_train_steps'] = \
        args.data['target_train_samples'] // args.data['train_batch_size']

    if args.model.get('seq_reducing') and \
            args.model.get('seq_reducing').get('enable_seq_reducing'):
        seq_len = args.model.get('seq_reducing').get('seq_len_list')
        args.model['seq_len'] = get_seq_len(seq_len)
    else:
        seq_len = args.model['seq_len']

    if df is None:
        traffic_df_path = Path(args.paths['traffic_df_filename'])
        if traffic_df_path.suffix in ['.h5', '.hdf5']:
            df = pd.read_hdf(args.paths['traffic_df_filename'])
            df.index.name = 'timestamp'
            if not traffic_df_path.with_suffix('.csv').exists():
                df.to_csv(traffic_df_path.with_suffix('.csv').__str__(), sep=',')
        else:
            sep = ',' if traffic_df_path.suffix in ['.csv'] else ' '
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

    test_samples = args.data.get('test_samples_per_epoch')
    val_samples = args.data.get('val_samples_per_epoch')
    assert test_samples >= 1
    assert val_samples >= 0

    train_batch_size = args.data['train_batch_size']
    val_batch_size = args.data['val_batch_size']
    test_batch_size = args.data['test_batch_size']

    assert test_samples >= test_batch_size, \
        'test_samples: {} | test_batch_size:{}'.format(test_samples, test_batch_size)
    assert val_samples == 0 or val_samples >= val_batch_size, \
        'val_samples: {} | val_batch_size:{}'.format(val_samples, val_batch_size)

    horizon = args.model['horizon']
    timesteps_per_sample = (sum(seq_len) if isinstance(seq_len, list) else seq_len) + horizon

    test_timesteps = test_samples - 1 + timesteps_per_sample
    val_timesteps = val_samples - 1 + timesteps_per_sample if val_samples > 0 else 0

    if val_timesteps == 0:
        logger.warning('Test dataset will be used as validation dataset as well. '
              'To use separate validation dataset, increase val_timesteps. ')

    source_length, num_nodes = df.shape

    test_length = test_timesteps if test_timesteps <= source_length else source_length
    val_length = val_timesteps if \
        (test_timesteps + val_timesteps) <= source_length else 0
    train_length = source_length - test_length - val_length

    length_dict = {'train_length': train_length,
                   'val_length': val_length,
                   'test_length': test_length}

    arr2d = df.values.astype(np.float32)
    arr2d_list = [arr2d]

    # Note: extract the fractional part (time) after reduce_mean
    days_arr1d = ((df.index - datetime(1970, 1, 1)).total_seconds() / \
                        timedelta(days=1).total_seconds()).astype(np.float32)

    features = args.get('supplemental_features')
    if features:
        if features.get('add_time_in_day'):
            arr2d_list.append(broadcast_last_dim(days_arr1d, arr2d.shape[-1]))
        if features.get('add_hist_dow_hour_mean'):
            arr2d_list.append(aggregate_keeping_rows(df, train_length).values.astype(np.float32))
        if features.get('add_day_of_week'):
            arr2d_list.append(broadcast_last_dim(days_arr1d, arr2d.shape[-1]))

    arr3d = np.stack(arr2d_list, axis=-1)
    num_samples, num_nodes, input_dim = arr3d.shape

    args.model['num_nodes'] = num_nodes
    args.model['input_dim'] = input_dim
    args.model['output_dim'] = 1

    dataloaders = setup_dataloader(
        arr3d,
        seq_len=seq_len,
        horizon=horizon,
        length_dict=length_dict,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        scale=args.data['scale'],
        features=features,
        logger=logger,
        )

    args.data['train_steps_per_epoch'] = \
        min(args.data['train_samples_per_epoch'] // args.data['train_batch_size'],
            dataloaders['train_loader'].num_batch)

    return args, dataloaders


def preprocess(args, show=True):
    for path_str in args.paths.values():
        parent_str = Path(path_str).parent.__str__() \
            if Path(path_str).suffix.__len__() > 0 \
            else path_str
        os.makedirs(parent_str, exist_ok=True)

    logger = config_logging(args)

    logger.info('Started preprocessing.')
    logger.info('Arguments read from the yaml file: \n' + pformat(args))

    args.logger = logger
    config_filepath = args.paths.get('config_filepath')
    if config_filepath:
        shutil.copy2(config_filepath, args.paths['model_dir'])

    args.timestep_size_freq = '{}min'.format(args.timestep_size_in_min)

    need_st_flag = not Path(args.paths['traffic_df_filename']).exists()
    need_geo_flag = not Path(args.paths['geohash6_filename']).exists()

    if need_st_flag or need_geo_flag:
        source_table_dir = Path(args.paths.get('source_table_dir'))
        if not source_table_dir.exists():
            raise FileNotFoundError('Directory not found: ' + args.paths.get('source_table_dir'))
        source_candidates = list(source_table_dir.glob('*.csv'))
        if not source_candidates:
            raise FileNotFoundError('No CSV file found at : ' + args.paths.get('source_table_dir'))
        if source_candidates.__len__() >= 2:
            logger.warning('Multiple CSV files found: {}'.format(source_candidates))
        source_table_filename = source_candidates[0].__str__()
        logger.info('Reading: {}'.format(source_table_filename))
        s_df = pd.read_csv(source_table_filename)

    if need_geo_flag:
        logger.info('Preparing geohash dataframe...')
        ##%% Prepare for adjacency matrix
        geo_df = get_geo_df(s_df)
        geo_df.to_csv(args.paths['geohash6_filename'], index=False)
        node_ids = geo_df['geohash6'].values
    else:
        geo_df = pd.read_csv(args.paths.get('geohash6_filename'), index_col=False)
        node_ids = geo_df['geohash6'].values

    st_df = None
    if need_st_flag:
        logger.info('Preparing spatio-temporal dataframe...')
        # Get Spatio-temporal df
        st_df = get_spatiotemporal_df(s_df, args, node_ids)
        st_df.to_csv(args.paths['traffic_df_filename'])
        logger.info('Spatio-temporal dataframe was saved at: {}'.format(args.paths['traffic_df_filename']))

    args, dataloaders = generate_train_val_test(args, st_df)

    adj_mx = get_adj_mat(args, geo_df, node_ids)

    logger.info('Completed preprocessing.')
    logger.info('Arguments after preprocessing: \n' + pformat(args))

    return args, dataloaders, adj_mx, node_ids


def transform_to_long(pred_df=None):
    # if pred_df is None:
    #     pred_df = pd.read_csv(args.paths['pred_df_filename'], parse_dates=[0], index_col=0)
    long_df = pd.DataFrame(pred_df.stack(), columns=['demand'])
    long_df.reset_index(inplace=True)
    long_df.loc[:, 'day'] = \
        long_df['timestamp'].apply(lambda x: int((x - datetime(1970, 1, 1)).days) + 1)
    long_df.loc[:, 'timestamp'] = long_df['timestamp'].apply(lambda x: x.strftime('%H:%M'))
    long_df.rename(columns={'level_1': 'geohash6'}, inplace=True)
    long_df = long_df[['geohash6', 'day', 'timestamp', 'demand']]
    return long_df


def save_pred_long_df(args, long_df):
    long_filename = args.paths['pred_long_filename']
    model_filename = args.paths['model_filename']
    long_filename = \
        Path(long_filename).parent / \
        '{}_{}'.format(Path(model_filename).name, Path(long_filename).name)
    long_df.to_csv(long_filename, index=False)
    print('The final prediction output file was saved at: {}'.format(long_filename))