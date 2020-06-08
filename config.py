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
    parser.add_argument('--optimizer', default='all', choices=['all', 'adam', 'sgd', 'sgd_to_half'],
                        help='Optimization algorithm to be used')
    parser.add_argument('--loss_function', default='all', choices=['all', 'cross_entropy', 'mse'],
                        help='Loss function to be used')
    parser.add_argument('--data', default='all', choices=['all', 'circle', 'mnist', 'fashion_mnist'])
    parser.add_argument('--silent', default=False, action='store_true')
    return parser.parse_args()


args = parse_args()
config = vars(args)
