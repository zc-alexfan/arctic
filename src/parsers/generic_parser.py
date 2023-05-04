def add_generic_args(parser):
    """
    Generic options that are non-specific to a project.
    """
    parser.add_argument("--agent_id", type=int, default=None)
    parser.add_argument(
        "--load_from", type=str, default=None, help="Load weights from InterHand format"
    )
    parser.add_argument(
        "--load_ckpt", type=str, default=None, help="Load checkpoints from PL format"
    )
    parser.add_argument(
        "--infer_ckpt", type=str, default=None, help="This is for the interface"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Resume training from checkpoint and keep logging in the same comet exp",
    )
    parser.add_argument(
        "-f",
        "--fast",
        dest="fast_dev_run",
        help="single batch for development",
        action="store_true",
    )
    parser.add_argument(
        "--trainsplit",
        type=str,
        default=None,
        choices=[None, "train", "smalltrain", "minitrain", "tinytrain"],
        help="Amount to subsample training set.",
    )
    parser.add_argument(
        "--valsplit",
        type=str,
        default=None,
        choices=[None, "val", "smallval", "tinyval", "minival"],
        help="Amount to subsample validation set.",
    )
    parser.add_argument(
        "--run_on",
        type=str,
        default=None,
        help="split for extraction",
    )
    parser.add_argument("--setup", type=str, default=None)

    parser.add_argument("--log_every", type=int, default=None, help="log every k steps")
    parser.add_argument(
        "--eval_every_epoch", type=int, default=None, help="Eval every k epochs"
    )
    parser.add_argument(
        "--lr_dec_epoch",
        type=int,
        nargs="+",
        default=None,
        help="Learning rate decay epoch.",
    )
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--lr_dec_factor", type=int, default=None, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=None, help="Learning rate decay factor"
    )
    parser.add_argument("--num_exp", type=int, default=None)
    parser.add_argument("--acc_grad", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--eval_on",
        type=str,
        default=None,
        choices=[None, "val", "test", "minival", "minitest"],
        help="Test mode set to eval on",
    )

    parser.add_argument("--mute", help="No logging", action="store_true")
    parser.add_argument("--no_vis", help="Stop visualization", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--cluster_node", type=str, default=None)
    parser.add_argument("--bid", type=int, default=None, help="log every k steps")
    return parser
