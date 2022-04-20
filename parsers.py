import argparse


def train_options():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        type=str,
        default="IQA",
        help="Experiment name"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=320,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate of Model (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=32,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(224, 224),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--crop-time",
        default=4,
        type=int,
        help='Number of crops per image (default: %(default)s)',
    )
    parser.add_argument(
        "--cuda",
        default=True,
        help="Use cuda"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=123.,
        help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm of Model (default: %(default)s",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="Path to a checkpoint"
    )
    parser.add_argument(
        "--backbone",
        default="convnext_tiny",
        type=str,
        help="Backbone model to use"
    )
    parser.add_argument(
        "--loss",
        default="cross_entropy",
        type=str,
        help="cross_entropy / mse"
    )
    parser.add_argument(
        "--csv_file_path",
        default=None,
        type=str
    )
    args = parser.parse_args()

    return args
