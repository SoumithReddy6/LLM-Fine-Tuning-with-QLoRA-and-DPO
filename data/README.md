# Data

The sample files are tiny fixtures for local tests and CI. Full training should use Hugging Face datasets such as `SALT-NLP/FLUE-FiQA` plus any approved OpenOrca-style instruction mixture.

Expected JSONL fields:

- `id`: stable example identifier.
- `text`: source financial statement or sentence.
- `label`: one of `negative`, `neutral`, `positive`.
- `prediction`: model output label for evaluation files.
- `confidence`: probability assigned to the predicted label.
- `latency_ms`: measured inference latency.
