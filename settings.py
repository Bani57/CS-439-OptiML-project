""" Module containing the implementation of the procedure required to parse the command line arguments """

import argparse


def parse_args():
    """
    Helper function to parse the command line arguments of the script in order to set the chosen program settings

    :returns: program settings, dictionary {command line argument: value}
    """

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_dir', default='./data/', help='Directory where datasets are stored.')
    parser.add_argument('--results_dir', default='./results/', help='Directory where log files are stored.')
    parser.add_argument('--plots_dir', default='./results/plots/', help='Directory where plots are stored.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for the computations.')
    parser.add_argument('--verbosity_mod', type=int, default=10, help="At what rate to print experiment progress.")
    return vars(parser.parse_args())


settings = parse_args()
