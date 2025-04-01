import argparse

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """

    # Session parameters
    parser.add_argument('--gpu_num',        type=int,   default=1,          help='GPU number to use')
    parser.add_argument('--batch_size',     type=int,   default=32,         help='Minibatch size')
    parser.add_argument('--num_workers',    type=int,   default=4,          help='Worker')
    parser.add_argument('--epochs',         type=int,   default=50,         help='Number of epochs to train')
    parser.add_argument('--print_every',    type=int,   default=1,          help='How many iterations print for loss evaluation')
    parser.add_argument('--save_every',     type=int,   default=1,          help='How many iterations to save')                        
    parser.add_argument('--eval_every',     type=int,   default=1,          help='How many iterations to save')

    # experimental setting
    parser.add_argument('--exp_name',       type=str,   default="baseline")

    # Learning Rate
    parser.add_argument('--lr',             type=float, default=1e-4,       help='learning rate, default=0.0005')

    # Directory parameters
    parser.add_argument('--data_dir',       type=str,   default='data',     help='dataset')
    parser.add_argument('--dataset_img',    type=str,   default='images',   help='image dataset')
    parser.add_argument('--dataset_label',  type=str,   default='labels',   help='label dataset')
    parser.add_argument('--weights',        type=str,   default="ckpt.pth", help='Weight Name')

    parser.add_argument('--save_dir',       type=str,   default="result",   help='Weight Name')
    parser.add_argument('--save_img',       type=str2bool, default=True,    help='save test image or not')

    # Design Paramters
    parser.add_argument('--image_size', nargs = '+', type=int, default=[128, 256], help='Image Size')

    # Architecture parameters
    parser.add_argument('--loss_type',      type=str,   default='l1',     choices=['l1', 'l2', 'ce'])

def str2bool(v):
    if isinstance(v, bool):
       return v
    elif v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))
