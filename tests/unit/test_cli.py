from click.testing import CliRunner

from precise_mrd.cli import main


def test_cli_help():
    """Test the CLI help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage: main [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Deterministic ctDNA/UMI MRD pipeline" in result.output

