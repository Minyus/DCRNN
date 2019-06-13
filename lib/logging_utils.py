from logging import basicConfig, FileHandler, StreamHandler
from logging import getLogger, Formatter

logger = getLogger('dcrnn')

from pathlib import Path


def config_logging(args, log_filename='info.log'):
    level = args.get('log_level', 'INFO')
    log_sep = '|'
    format_str = log_sep.join(['%(asctime)s',
                               # '%(module)s',
                               # '%(funcName)s',
                               '%(levelname)s',
                               '%(message)s'])

    logger.setLevel(level)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    logger.addHandler(stream_handler)

    if args.get('paths'):
        log_dir_path = args.get('paths').get('model_dir')
        if log_dir_path:
            log_file_path = (Path(log_dir_path) / log_filename).__str__()
            file_handler = FileHandler(log_file_path)
            file_handler.setFormatter(Formatter(format_str))
            logger.addHandler(file_handler)


