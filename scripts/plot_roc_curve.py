import sys
import pathlib
import argparse

sys.path.append(pathlib.Path.cwd())

from common import read_csv_results, plot_roc 


def parse_args():
    parser = argparse.ArgumentParser('Plot ROC curve from csv file.')
    parser.add_argument('results', help='path to the results csv file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # parse parameters in the log
    results_csv = pathlib.Path(args.results)

    perf = read_csv_results(results_csv)

    plot_roc(perf, results_csv.parent / 'roc.png')

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
