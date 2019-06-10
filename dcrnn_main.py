import sys
import os

from lib.utils import read_yaml
from lib.preprocess import preprocess, transform_to_long
from model.dcrnn_top import train_dcrnn, run_dcrnn


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    args = read_yaml('dcrnn_config.yaml')
    args, dataloaders, adj_mx, node_ids = preprocess(args)
    train_dcrnn(args, dataloaders, adj_mx)
    pred_df = run_dcrnn(args, dataloaders, adj_mx, node_ids)
    long_df = transform_to_long(pred_df)
    long_df.to_csv(args.paths['pred_long_filename'], index=False)
    print('The final prediction output file was saved at: \n >> {}'.\
          format(args.paths['pred_long_filename']))
