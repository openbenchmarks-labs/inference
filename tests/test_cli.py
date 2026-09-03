from openbenchmarks_inference.cli import main
from openbenchmarks_inference.workflows.micro_single_task import load_config


def test_cli_lists_the_registered_workflow(capsys):
    assert main(["list-workflows"]) == 0
    assert capsys.readouterr().out == "micro-single-task\n"


def test_bundled_workflow_spec_is_packaged_and_loadable():
    config = load_config()
    assert config.benchmark_slug == "inference-micro-single-task"
    assert len(config.providers) == 10
