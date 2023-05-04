from loguru import logger


def set_default_params(args, default_args):
    # if a val is not set on argparse, use default val
    # else, use the one in the argparse
    custom_dict = {}
    for key, val in args.items():
        if val is None:
            args[key] = default_args[key]
        else:
            custom_dict[key] = val

    logger.info(f"Using custom values: {custom_dict}")
    return args
