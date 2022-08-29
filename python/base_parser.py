import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Set the seed for RandomState',
    )
    parser.add_argument(
        '--d',
        dest='d',
        type=int,
        default=1,
        help='Set the dimension d. Default: 1',
    )
    parser.add_argument(
        '--alpha-i',
        dest='alpha_i',
        type=float,
        default=1.,
        help='Set barrier height of the i-th coordinate for the multidimensional extension \
              of the double well potential. Default: 1.',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=1.,
        help='Set the beta parameter. Default: 1.',
    )
    parser.add_argument(
        '--xzero-i',
        dest='xzero_i',
        type=float,
        default=-1.,
        help='Set the initial posicion of the process at each axis. Default: -1',
    )
    parser.add_argument(
        '--h',
        dest='h',
        type=float,
        default=0.1,
        help='Set the discretization step size. Default: 0.1',
    )
    parser.add_argument(
        '--K',
        dest='K',
        type=int,
        default=10**3,
        help='Set number of trajectories to sample. Default: 1.000',
    )
    parser.add_argument(
        '--dt',
        dest='dt',
        type=float,
        default=0.01,
        help='Set dt. Default: 0.01',
    )
    parser.add_argument(
        '--k-lim',
        dest='k_lim',
        type=int,
        default=10**8,
        help='Set maximal number of time steps. Default: 100.000.000',
    )
    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        choices=['sgd', 'adam'],
        default='adam',
        help='Set type of optimizer. Default: "adam"',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.01,
        help='Set learning rate. Default: 0.01',
    )
    parser.add_argument(
        '--n-iterations',
        dest='n_iterations',
        type=int,
        default=100,
        help='Set maximal number of sgd iterations. Default: 100',
    )
    parser.add_argument(
        '--n-iterations-backup',
        dest='n_iterations_backup',
        type=int,
        help='Set number of sgd iterations between saving the arrays. Default: None',
    )
    parser.add_argument(
        '--do-u-l2-error',
        dest='do_u_l2_error',
        action='store_true',
        help='compute u l2 error. Default: False',
    )
    parser.add_argument(
        '--n-layers',
        dest='n_layers',
        type=int,
        default=3,
        help='Set total number of layers. Default: 3',
    )
    parser.add_argument(
        '--d-hidden-layer',
        dest='d_hidden_layer',
        type=int,
        default=30,
        help='Set dimension of the hidden layers. Default: 30',
    )
    return parser
