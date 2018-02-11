import argparse

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
        required=True
    )         
    parser.add_argument(
        '--num_of_epochs',
        help='Epochs for training',
        required=True
    )  

    # parse args
    args = parser.parse_args()
    arguments = args.__dict__
    print(arguments)
    traindata = arguments.pop('traindata')
    print(arguments)
