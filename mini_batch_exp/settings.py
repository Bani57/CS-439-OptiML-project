import argparse

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', default='./data/', help='Directory where data are stored')
    parser.add_argument('--logs_dir', default='./logs/', help='Directory where log files are stored')
    parser.add_argument('--models_dir', default='./models/', help='Directory where trained models are stored')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of available GPUs')
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of the GPU to use")
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--split_task', default=False, action='store_true',
            help='Execute only a subset of the experiments based on the GPU id. Can be used to split tasks betwenn GPUs.')
    parser.add_argument('--verbosity_mod', type=int, default=10)
    parser.add_argument('--hide_settings', default=False, action='store_true')
    return parser.parse_args()

args = parse_args()
settings = vars(args)