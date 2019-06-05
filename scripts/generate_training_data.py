from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path

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
        time_ind = (df.index.values - \
                    df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
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
    traffic_df_path = Path(args.traffic_df_filename)
    if traffic_df_path.suffix in ['.h5', '.hdf5']:
        df = pd.read_hdf(args.traffic_df_filename)
        df.index.name = 'timestamp'
        if not traffic_df_path.with_suffix('.csv').exists():
            df.to_csv(traffic_df_path.with_suffix('.csv').__str__(), sep=',')
    else:
        df = pd.read_csv(args.traffic_df_filename, index_col=0, parse_dates=[0])


    # 0 is the latest observed sample.
    # x_offsets = np.sort(
    #     # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
    #     np.concatenate((np.arange(-args.history_timesteps + 1, 1, 1),))
    # )
    x_offsets = np.arange(-args.history_timesteps + 1, 1, 1)
    # Predict the next one hour
    # y_offsets = np.sort(np.arange(1, args.future_timesteps + 1, 1))
    y_offsets = np.arange(1, args.future_timesteps + 1, 1)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
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

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
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
    parser.add_argument("--test_timesteps", type=int, default=6850,
                        help="timesteps for test",)
    parser.add_argument("--validation_timesteps", type=int, default=3425,
                        help="timesteps for validation",)
    args = parser.parse_args()
    main(args)
