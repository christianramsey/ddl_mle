import argparse
import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--traindata',
        help='Training data directory or files',
        required=True
    )
    parser.add_argument(
        '--evaldata',
        help='Eval data directory or files',
        required=True
    )
    parser.add_argument(
        '--batchsize',
        help='Batch size for training',
        required=True,
        type=int,
        default=256
    )         
    parser.add_argument(
        '--epochs',
        help='Epochs for training',
        required=True
    )  
    parser.add_argument(
        '--jobdir',
        help='Job dir for ml engine',
        required=True
    )      

    # parse args
    args = parser.parse_args()
    arguments = args.__dict__
    print(arguments)
    jobdir = arguments.pop('jobdir')
    print(arguments)

    model.train_eval(**arguments)