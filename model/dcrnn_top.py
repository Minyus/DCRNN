from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from model.dcrnn_supervisor import DCRNNSupervisor


def get_model_filename(args):
    dir = args.paths['model_dir']
    path_list = list(Path(dir).glob('*.index'))
    if path_list:
        global_step = max([int(p.stem.split('-')[-1]) for p in path_list])
        path = Path(dir).glob('*{}.index'.format(global_step)).__next__()
        model_filename = (path.parent / path.stem).as_posix()
    else:
        model_filename = None
        global_step = 0
    args.train['global_step'] = global_step
    args.paths['model_filename'] = model_filename
    return args


def setup_tf(args):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    return tf_config


def train_dcrnn(args, dataloaders, adj_mx):
    if not args.test_only:
        args = get_model_filename(args)
        tf_config = setup_tf(args)
        with tf.Session(config=tf_config) as sess:
            supervisor = \
                DCRNNSupervisor(sess, adj_mx=adj_mx, dataloaders=dataloaders, **args)
            # model_filename = args.paths['model_filename']
            # if model_filename:
            #     supervisor.load(sess, model_filename)
            supervisor.train(sess=sess)
    return args


def run_dcrnn(args, dataloaders, adj_mx, node_ids):
    # logger = utils.get_logger(args.paths['model_dir'], __name__, level=args.get('log_level', 'INFO'))
    args = get_model_filename(args)
    model_filename = args.paths['model_filename']
    pred_df = None
    if model_filename:
        tf_config = setup_tf(args)
        with tf.Session(config=tf_config) as sess:
            supervisor = \
                DCRNNSupervisor(sess, adj_mx=adj_mx, dataloaders=dataloaders, **args)
            # supervisor.load(sess, model_filename)
            outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.paths['output_filename'], **outputs)

        pred_tensor = np.stack(outputs['predictions'])
        # pred_arr2d = pred_tensor[:, -1, :]
        pred_arr2d = pred_tensor[:, 0, :]  # Note: the indices are reversed
        np.savetxt(args.paths['pred_arr2d_filename'], pred_arr2d, delimiter=',')
        # print('Predictions saved as {}.'.format(args.paths['pred_arr2d_filename']))

        # pred_df = pd.read_csv(args.paths['pred_arr2d_filename'], index_col=False,
        #                       sep=',', header=None)
        pred_df = pd.DataFrame(pred_arr2d)
        pred_df.columns = node_ids
        pred_df['timestamp'] = \
            pd.date_range(start=args.datetime_future_start, periods=args.model['horizon'], freq=args.timestep_size_freq)
        pred_df = pred_df.set_index('timestamp')
        pred_df.to_csv(args.paths['pred_df_filename'])
    else:
        print('Pretrained model was not found in the directory: {}'.\
              format(args.paths['model_dir']))
    return args, pred_df






