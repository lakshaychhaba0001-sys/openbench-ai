"""Run the custom metric tutorial's snippets against the local CLI."""

from __future__ import annotations

import configparser
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC = "tutorial_normalized_match"


def test_custom_metric_tutorial(tmp_path):
    doc = (ROOT / "docs" / "tutorial_custom_metric.md").read_text(encoding="utf-8")
    files = re.findall(r'```(?:python|toml) title="([^"]+)"\n(.*?)\n```', doc, re.S)
    for name, source in files:
        (tmp_path / name).write_text(source + "\n", encoding="utf-8")

    # Expose the documented entry point with the same distribution metadata
    # that an installation would provide. Keep the test independent of pip and
    # the package index.
    config = configparser.ConfigParser()
    config.read(tmp_path / "pyproject.toml", encoding="utf-8")
    project = config["project"]
    name = json.loads(project["name"])
    version = json.loads(project["version"])
    target = json.loads(config['project.entry-points."openbench.metrics"'][METRIC])
    distribution = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    (distribution / "entry_points.txt").write_text(
        f"[openbench.metrics]\n{METRIC} = {target}\n", encoding="utf-8"
    )

    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT / "src")]))
    environment.pop("OPENBENCH_DISABLE_PLUGINS", None)
    for block in re.findall(r"```bash\n(.*?)\n```", doc, re.S):
        for line in block.splitlines():
            program, *args = shlex.split(line)
            if program == "python" and args[:3] == ["-m", "pip", "install"]:
                continue
            prefix = [sys.executable, "-m", "openbench.cli"] if program == "openbench" else [sys.executable]
            result = subprocess.run(
                [*prefix, *args], cwd=tmp_path, env=environment, text=True,
                capture_output=True, timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            if program == "openbench" and args == ["metrics", "list"]:
                assert METRIC in result.stdout

    def report(name):
        return json.loads((tmp_path / name).read_text(encoding="utf-8"))

    assert report("baseline.json")["aggregate_results"][METRIC]["score"] == 0.666667
    assert report("challenger.json")["aggregate_results"][METRIC]["score"] == 1.0
    assert report("comparison.json")["provider_calls"] == 0
    assert '"score_kind": "binary"' in json.dumps(report("comparison.json"))
    assert report("ranking.json")["metric"] == METRIC
    assert report("ranking.json")["score_kind"] == "binary"
