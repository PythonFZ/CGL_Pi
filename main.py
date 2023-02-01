"""Command Line Interface for pi estimators."""

import click

import pi

estimator = {
    "MonteCarlo": pi.MonteCarloPiEstimator,
    "BaileyBorweinPlouffe": pi.BaileyBorweinPlouffeEstimator,
    "Leibniz": pi.LeibnizPiEstimator,
}


@click.command()
@click.option("--method", help=f"Select from {', '.join(estimator)}.")
@click.option("-n", help="Argument for the selected pi estimator method.")
def cli(method: str, n: int) -> None:
    """Estimate pi using various methods."""
    method = estimator[method]
    click.echo(f"{method.__name__}: {method(int(n)).estimate()}")


if __name__ == "__main__":
    cli()
