import argparse
import logging
import os

if __name__ == '__main__':
    ################################### Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs",type=int  , default=10)
    parser.add_argument("--lr"        ,type=float, default=0.01)
    parser.add_argument("--s3_path"   ,type=str)
    args = parser.parse_args()
    ################################### Logging

    assert(os.path.isdir(args.s3_path))

    
    logging_path = os.path.join(args.s3_path,"logs","logger.log")
    logging.basicConfig(filename = logging_path,
                        format   = '%(asctime)s;%(message)s',
                        filemode = "w",
                        level    = logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.error("Harmless debug Message")