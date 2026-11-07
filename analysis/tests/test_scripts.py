"""Shell scripts in scripts/ must at least parse (bash -n) and keep their key settings."""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_slurm_script_parses():
    r = subprocess.run(["bash", "-n", str(SCRIPTS / "slurm_train.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_slurm_script_settings():
    s = (SCRIPTS / "slurm_train.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=96:00:00" in s
    assert "--signal=B:USR1@" in s
    assert "--stop-file" in s and "--run-dir" in s
    assert "sbatch --dependency=afterany" in s
