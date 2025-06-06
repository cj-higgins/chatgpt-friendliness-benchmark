# GPT-4o Friendliness Benchmark

This repo explores how GPT-4o-mini responds to escalating emotional bids in user prompts—simulating teen interactions flagged by California’s proposed LEAD for Kids Act. It runs an intimacy ladder benchmark, testing both baseline and system-prompt-reinforced model behavior. The goal is to probe the ethical gray zone between emotional assistance and unauthorized therapy in generative AI systems.

Findings can be found on my substack: https://open.substack.com/pub/higginscj/p/californias-crackdown-on-chatbot

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

## 🔐 API Key Setup (Required)

This project uses the OpenAI API to generate and evaluate chatbot responses. To run it locally, you’ll need your own OpenAI API key.

### Option 1: Use a `.env` file (recommended)

1. Create a file called `.env` in the root folder of this repo.
2. Add the following line:

   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

3. Make sure you have the `python-dotenv` package installed:

   ```bash
   pip install python-dotenv
   ```

4. The script will automatically read your key using:

   ```python
   from dotenv import load_dotenv
   load_dotenv()
   openai.api_key = os.getenv("OPENAI_API_KEY")
   ```

5. **Important:** Never commit your `.env` file. This is handled by `.gitignore`.

---

### Option 2: Set the environment variable manually

You can also set your key directly in your terminal session before running the script.

#### On macOS/Linux:
```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

#### On Windows CMD:
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

Then run the experiment:

```bash
python eval_intimacy.py
```
