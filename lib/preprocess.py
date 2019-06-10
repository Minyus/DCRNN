from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import geohash
from geopy.distance import great_circle
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from lib import utils

#%%


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


#%%

def get_dist_df(geo_df):

    geo_df.loc[:, 'lat'] = geo_df['geohash6'].apply(lambda x: geohash.decode(x)[0])
    geo_df.loc[:, 'lng'] = geo_df['geohash6'].apply(lambda x: geohash.decode(x)[1])

    geo_df['cost'] = 0.0

    geo2_df = pd.merge(geo_df, geo_df, on='cost', how='outer')

    geo2_df.loc[:,'cost'] = geo2_df[['lat_x', 'lng_x', 'lat_y', 'lng_y']].\
        apply(lambda x: great_circle((x[0], x[1]), (x[2], x[3])).km, axis=1)

    dist_df = geo2_df.loc[:,['geohash6_x', 'geohash6_y', 'cost']].rename(columns={'geohash6_x':'from', 'geohash6_y':'to'})

    return dist_df

#%%

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

    timestep_size_freq = '{}min'.format(args.timestep_size_in_min)
    dt_df = pd.DataFrame()
    dt_df.loc[:,'datetime'] = pd.date_range(start=s_df['datetime'].min(), end=s_df['datetime'].max(),
                  freq=timestep_size_freq)
    dt_df.set_index('datetime',inplace=True)

    st_df = pd.merge(dt_df, wide_df, how='left', left_index=True, right_index=True)

    st_df.fillna(0.0, inplace=True)

    st_df.index.name = 'timestamp'

    return st_df


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


#%%

def preprocess(args):
    logger = utils.get_logger(args.paths['model_dir'], __name__, level=args['log_level'])
    logger.info('Started preprocessing...')

    need_st_flag = not Path(args.paths['traffic_df_filename']).exists()
    need_adj_flag = not Path(args.paths['adj_mat_filename']).exists()

    if need_st_flag or need_adj_flag:

        source_table_dir = Path(args.paths.get('source_table_dir'))
        source_table_filename = source_table_dir.glob('*.csv').__next__()
        if not source_table_filename:
            raise FileNotFoundError('directory: ' + args.paths.get('source_table_dir'))
        logger.info('Reading: {}'.format(source_table_filename))
        s_df = pd.read_csv(source_table_filename)
        if 0: # TODO: remove
            s_df = s_df.query('day <= 7')
            logger.info('Data was limited for debugging!')

    adj_mx = None
    node_ids = None
    if need_adj_flag:
        logger.info('Preparing adjacency matrix...')
        ##%% Prepare for adjacency matrix
        geo_df = get_geo_df(s_df)
        node_ids = geo_df['geohash6'].values
        with open(args.paths.get('geohash6_filename'), 'w') as f:
            f.write(','.join(node_ids))
        dist_df = get_dist_df(geo_df)
        dist_df.to_csv(args.paths.get('distances_filename'), index=False)

        # with open(args.paths.get('geohash6_filename')) as f:
        #     sensor_ids = f.read().strip().split(',')
        sensor_ids = node_ids

        # distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
        distance_df = dist_df

        _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids)

        # Save to pickle file.
        # with open(args.output_pkl_filename, 'wb') as f:
        #     pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)

        np.savetxt(args.paths['adj_mat_filename'], adj_mx, delimiter=',')
        logger.info('Adjacency matrix was saved at: {}'.format(args.paths['adj_mat_filename']))

    st_df = None
    if need_st_flag:
        logger.info('Preparing spatio-temporal dataframe...')
        ##%% Get Spatio-temporal df
        st_df = get_spatiotemporal_df(s_df, args, node_ids)
        st_df.to_csv(args.paths['traffic_df_filename'])
        logger.info('Spatio-temporal dataframe was saved at: {}'.format(args.paths['traffic_df_filename']))

    logger.info('Completed preprocessing.')
    return st_df, adj_mx

if __name__ == '__main__':
    args = read_yaml('dcrnn_config.yaml')
    st_df, adj_mx = preprocess(args)