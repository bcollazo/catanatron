import os
import tempfile

from click.testing import CliRunner
import pandas as pd

from catanatron.cli.play import simulate


def test_play():
    runner = CliRunner()
    result = runner.invoke(simulate, ["--num=5", "--players=R,F,VP,W"])
    assert result.exit_code == 0
    assert "Game Summary" in result.output


def test_play_strong():
    runner = CliRunner()
    result = runner.invoke(simulate, ["--num=1", "--players=AB,SAB,M:3:True,G:3"])
    assert result.exit_code == 0
    assert "Game Summary" in result.output


def test_csv_play():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = runner.invoke(
            simulate, ["--num=1", "--players=F,F", "--csv", "--output", tmpdirname]
        )
        assert result.exit_code == 0

        # Assert 5 gzipped dataframes were created in tmpdirname
        assert len(os.listdir(tmpdirname)) == 5
        # Assert they have the correct dimensions
        actions_df = pd.read_csv(
            os.path.join(tmpdirname, "actions.csv.gzip"), compression="gzip"
        )
        board_tensors_df = pd.read_csv(
            os.path.join(tmpdirname, "board_tensors.csv.gzip"), compression="gzip"
        )
        main_df = pd.read_csv(
            os.path.join(tmpdirname, "main.csv.gzip"), compression="gzip"
        )
        rewards_df = pd.read_csv(
            os.path.join(tmpdirname, "rewards.csv.gzip"), compression="gzip"
        )
        samples_df = pd.read_csv(
            os.path.join(tmpdirname, "samples.csv.gzip"), compression="gzip"
        )
        num_samples = len(samples_df)

        assert len(actions_df) == num_samples
        assert len(board_tensors_df) == num_samples
        assert len(main_df) == num_samples
        assert len(rewards_df) == num_samples
