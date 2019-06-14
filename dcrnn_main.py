import sys
import os
import logging

from lib.utils import read_yaml
from lib.preprocess import preprocess, transform_to_long, save_pred_long_df
from model.dcrnn_top import train_dcrnn, run_dcrnn


sys.path.append(os.getcwd())
args = read_yaml('dcrnn_config.yaml')
args, dataloaders, adj_mx, node_ids = preprocess(args)
args = train_dcrnn(args, dataloaders, adj_mx)
args, pred_df = run_dcrnn(args, dataloaders, adj_mx, node_ids)
long_df = transform_to_long(pred_df)
save_pred_long_df(args, long_df)
logging.shutdown()
