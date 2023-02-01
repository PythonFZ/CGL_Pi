"""Test the pi module."""
import subprocess

import click.testing
import numpy as np
import pytest

import main
import pi


def test_PiEstimator() -> None:
    """Test abstract base class PiEstimator."""
    with pytest.raises(TypeError):
        pi.PiEstimator()


def test_MCPiEstimator() -> None:
    """Test the Monte Carlo PiEstimator."""
    assert pi.MonteCarloPiEstimator(1000000).estimate() == pytest.approx(np.pi, 0.01)


def test_MCPiEstimator_n_error() -> None:
    """Test the Monte Carlo PiEstimator with wrong input type."""
    with pytest.raises(TypeError):
        _ = pi.MonteCarloPiEstimator(10e5).estimate()


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_BaileyBorweinPlouffe(n: int) -> None:
    """Test the BaileyBorweinPlouffe Estimator."""
    assert pi.BaileyBorweinPlouffeEstimator(n).estimate() == pytest.approx(np.pi, 0.001)


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_LeibnizPiEstimator(n: int) -> None:
    """Test the Leibniz PiEstimator."""
    assert pi.LeibnizPiEstimator(n).estimate() == pytest.approx(np.pi, 0.01)


def test_cli() -> None:
    """Test the command line interface."""
    result = subprocess.run(
        ["python", "main.py", "--method", "MonteCarlo", "-n", "1000000"],
        capture_output=True,
        text=True,
    )

    method, estimate = result.stdout.split(": ")

    assert method == "MonteCarloPiEstimator"
    assert float(estimate) == pytest.approx(np.pi, 0.01)


def test_cli_directly() -> None:
    """Test the command line interface via CLIRunnner."""
    runner = click.testing.CliRunner()
    result = runner.invoke(main.cli, ["--method", "MonteCarlo", "-n", "1000000"])
    assert result.exit_code == 0
    method, estimate = result.output.split(": ")

    assert method == "MonteCarloPiEstimator"
    assert float(estimate) == pytest.approx(np.pi, 0.01)
