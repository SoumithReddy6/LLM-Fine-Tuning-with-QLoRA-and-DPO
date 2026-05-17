from pathlib import Path

from qlora_dpo_finance.config import load_config


def test_load_config_reads_experiment_name():
    config = load_config(Path("configs/experiments/eval_financial_sentiment.yaml"))

    assert config.name == "eval_financial_sentiment"
    assert config.task == "financial_sentiment"
    assert config.section("evaluation")["predictions_path"].endswith("predictions_sample.jsonl")
