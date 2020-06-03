import argparse


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', default='./data/', help='Directory where datasets are stored')
    parser.add_argument('--results_dir', default='./results/', help='Directory where log files are stored')
    parser.add_argument('--plots_dir', default='./results/plots/', help='Directory where plots are stored')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for the computations')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of available GPUs')
    parser.add_argument('--split_task', default=True, action='store_true',
                        help='Execute only a subset of the experiments based on the GPU id.'
                             'Can be used to split tasks between GPUs.')
    parser.add_argument('--verbosity_mod', type=int, default=10)
    parser.add_argument('--hide_settings', default=False, action='store_true')
    return parser.parse_args()


args = parse_args()
settings = vars(args)
