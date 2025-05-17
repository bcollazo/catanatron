from click.testing import CliRunner

from catanatron.cli.play import simulate


def test_play():
    runner = CliRunner()
    result = runner.invoke(simulate, ["--num=5", "--players=R,V,VP,W"])
    assert result.exit_code == 0
    assert "Game Summary" in result.output


def test_play_strong():
    runner = CliRunner()
    result = runner.invoke(simulate, ["--num=1", "--players=AB,SAB,M:3:True,G:3"])
    assert result.exit_code == 0
    assert "Game Summary" in result.output
