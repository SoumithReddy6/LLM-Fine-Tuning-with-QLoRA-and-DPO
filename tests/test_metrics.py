from qlora_dpo_finance.metrics import evaluate_predictions, parse_prediction


def test_parse_prediction_accepts_single_label_text():
    assert parse_prediction("The label is positive") == "positive"
    assert parse_prediction("buy") is None


def test_evaluate_predictions_reports_core_metrics_and_lift():
    rows = [
        {"label": "positive", "prediction": "positive", "confidence": 0.9, "latency_ms": 10},
        {"label": "negative", "prediction": "negative", "confidence": 0.8, "latency_ms": 20},
        {"label": "neutral", "prediction": "positive", "confidence": 0.6, "latency_ms": 30},
    ]
    baseline = [
        {"label": "positive", "prediction": "neutral", "confidence": 0.4, "latency_ms": 100},
        {"label": "negative", "prediction": "negative", "confidence": 0.8, "latency_ms": 100},
        {"label": "neutral", "prediction": "positive", "confidence": 0.6, "latency_ms": 100},
    ]

    metrics = evaluate_predictions(rows, baseline)

    assert metrics["accuracy"] == 0.6667
    assert metrics["invalid_label_rate"] == 0.0
    assert metrics["relative_accuracy_lift"] > 0
    assert "macro_f1" in metrics
    assert "expected_calibration_error" in metrics
