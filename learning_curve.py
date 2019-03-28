import argparse
from utils import learning_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args()

    log_file = args.log_file

    learning_curve(log_file)


if __name__ == '__main__':
    main()