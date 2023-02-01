"""Command Line Interface for pi estimators."""

import argparse

import pi

estimator = {
    "MonteCarlo": pi.MonteCarloPiEstimator,
    "BaileyBorweinPlouffe": pi.BaileyBorweinPlouffeEstimator,
    "Leibniz": pi.LeibnizPiEstimator,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", help=f"Select from {', '.join(estimator)}.", required=True
)
parser.add_argument(
    "-n", help="Argument for the selected pi estimator method.", required=True
)

if __name__ == "__main__":

    args = parser.parse_args()
    method = estimator[args.method]
    print(f"{method.__name__}: {method(int(args.n)).estimate()}")
