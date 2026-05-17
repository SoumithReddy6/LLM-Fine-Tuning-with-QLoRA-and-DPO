# Colab / Kaggle Quickstart

1. Create a GPU runtime.
2. Clone the GitHub repository.
3. Install dependencies with `pip install -r requirements.txt`.
4. Login to Hugging Face with `huggingface-cli login`.
5. Run `accelerate config` and choose a single GPU.
6. Launch SFT with `accelerate launch scripts/train_sft.py --config configs/experiments/qwen25_7b_qlora_rank16.yaml`.
7. Launch DPO with `accelerate launch scripts/train_dpo.py --config configs/experiments/qwen25_7b_dpo.yaml`.
8. Run evaluation with `python3 scripts/evaluate_model.py --config configs/experiments/eval_financial_sentiment.yaml`.

For Llama 3, request model access on Hugging Face before launching the run.
