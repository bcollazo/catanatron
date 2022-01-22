from click.testing import CliRunner

from catanatron_experimental.play import simulate


def test_play():
    runner = CliRunner()
    result = runner.invoke(simulate, ["--num=5", "--players=R,R,R,W"])
    assert result.exit_code == 0
    assert "Game Summary" in result.output
