import sys
import os

from lib.utils import read_yaml
from lib.preprocess import preprocess
from model.dcrnn_top import train_dcrnn, run_dcrnn


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    args = read_yaml('dcrnn_config.yaml')
    args, dataloaders, adj_mx, node_ids = preprocess(args)
    train_dcrnn(args, dataloaders, adj_mx)
    run_dcrnn(args, dataloaders, adj_mx, node_ids)
