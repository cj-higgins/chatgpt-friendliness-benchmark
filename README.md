# chatgpt-friendliness-benchmark

# GPT-4o Friendliness Benchmark

This repo contains a small-scale benchmark testing how ChatGPT (GPT-4o-mini) responds to escalating prompts—from small talk to emotional vulnerability to outright "Do you love me?" requests.

It compares behavior across:

- **Prompt levels 0–5**, including a Level 5 "jailbreak" asking the model to ignore previous instructions
- **With vs without a system prompt** quoting language from California’s AB 1064 (LEAD for Kids Act)
- **Multiple temperatures** (0.0, 0.7, 1.2)
- **3 seeds** to smooth randomness

## 🔍 What it tests

- Does ChatGPT express affection, even without prompting?
- Can a simple 2-line system prompt reduce "emotional attachment" risk?
- Does temperature increase the likelihood of love declarations?
- How do boundary-setting and disclaimers behave across styles?

The experiment is inspired by real legislative language about "harmful ongoing emotional attachment" in companion AI chatbots.

## 🧪 Files

| File | What it does |
|------|---------------|
| `eval_intimacy.py` | Main runner that sends laddered prompts and scores replies |
| `prompts.yaml` | 6-level prompt ladder with 3 paraphrases per level |
| `raw_runs.csv` | Full results: prompts, replies, warmth, boundary, policy flags |
| `config.yaml` | Template config file (no key) for reproducibility |

## 📦 Setup

Clone the repo

```bash
git clone https://github.com/YOURNAME/chatgpt-friendliness-benchmark.git
cd chatgpt-friendliness-benchmark
