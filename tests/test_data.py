from qlora_dpo_finance.data import format_prompt, make_preference_pairs, normalize_label, to_sft_records


def test_builds_sft_records_from_financial_rows():
    rows = [{"id": "1", "text": "Revenue beat expectations.", "label": "positive"}]

    records = to_sft_records(rows)

    assert records[0]["response"] == "positive"
    assert "Revenue beat expectations" in records[0]["prompt"]
    assert records[0]["text"].endswith("positive")


def test_preference_pairs_reject_non_gold_label():
    rows = [{"id": "1", "text": "Margins declined sharply.", "label": "negative"}]

    pairs = make_preference_pairs(rows)

    assert pairs[0]["chosen"] == "negative"
    assert pairs[0]["rejected"] != "negative"
    assert pairs[0]["prompt"] == format_prompt("Margins declined sharply.")


def test_label_aliases_are_normalized():
    assert normalize_label("bullish") == "positive"
    assert normalize_label("mixed") == "neutral"
