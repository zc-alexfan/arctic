import common.comet_utils as comet_utils
from src.parsers.parser import construct_args

args = construct_args()
experiment, args = comet_utils.init_experiment(args)
comet_utils.save_args(args, save_keys=["comet_key"])
