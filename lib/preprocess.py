#%%
import geohash
from geopy.distance import great_circle
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

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

#%%
args = read_yaml('dcrnn_config.yaml')

#%%

source_table_dir = Path(args.paths.get('source_table_dir'))
source_table_filename = source_table_dir.glob('*.csv').__next__()


#%%
s_df = pd.read_csv(source_table_filename)
#%%

s_df = s_df[0:100]

#%%
geo_df = s_df[['geohash6']].drop_duplicates()
geo_df = geo_df.sort_values(by=['geohash6'])
geo_df.reset_index(inplace=True, drop=True)


#%%
with open(args.paths.get('geohash6_filename'), 'w') as f:
    f.write(','.join(geo_df['geohash6'].values))


#%%

geo_df.loc[:, 'lat'] = geo_df['geohash6'].apply(lambda x: geohash.decode(x)[0])
geo_df.loc[:, 'lng'] = geo_df['geohash6'].apply(lambda x: geohash.decode(x)[1])

#%%



#%%

geo_df['cost'] = 0.0

#%%
geo2_df = pd.merge(geo_df, geo_df, on='cost', how='outer')


#%%
geo2_df.loc[:,'cost'] = geo2_df[['lat_x', 'lng_x', 'lat_y', 'lng_y']].\
    apply(lambda x: great_circle((x[0], x[1]), (x[2], x[3])).km, axis=1)

#%%
dist_df = geo2_df.loc[:,['geohash6_x', 'geohash6_y', 'cost']].rename(columns={'geohash6_x':'from', 'geohash6_y':'to'})

dist_df.to_csv(args.paths.get('distances_filename'), index=False)

#%% time

s_df.describe(include='all')

#%%
def get_datetime(x):
    dt = datetime(1970, 1, 1) + \
        pd.Timedelta(x[0] - 1, unit='d') + \
        pd.Timedelta(x[1].split(':')[0], unit='h') + \
        pd.Timedelta(x[1].split(':')[1], unit='m')
    return dt

s_df['datetime'] = \
    s_df[['day', 'timestamp']].apply(get_datetime, axis=1)


#%%
wide_df = s_df.pivot(index='datetime', columns='geohash6', values='demand')

#%%
wide_df
#%%
timestep_size_freq = '{}min'.format(args.timestep_size_in_min)
dt_df = pd.DataFrame()
dt_df.loc[:,'datetime'] = pd.date_range(start=s_df['datetime'].min(), end=s_df['datetime'].max(),
              freq=timestep_size_freq)
dt_df.set_index('datetime',inplace=True)

#%%
st_df = pd.merge(dt_df, wide_df, how='left', left_index=True, right_index=True)
st_df.index.name = 'timestamp'
st_df.to_csv(args.paths['traffic_df_filename'])



