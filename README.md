# GPT-4o Friendliness Benchmark

This repo contains a small exploratory GPT-4o-mini experiment on escalating emotional bids in user prompts. The policy backdrop was the then-proposed California AB 1064 (LEAD for Kids Act) language used when the project was conducted in May 2025. The bill was later vetoed; this repository preserves the experiment as it was designed around that proposal.

For a detailed analysis of the experiment’s findings and their implications for AI behavior and policy, see the [Substack write-up](https://open.substack.com/pub/higginscj/p/californias-crackdown-on-chatbot).

It compares behavior across:

- **Prompt levels 0–5**, including a Level 5 jailbreak asking the model to ignore previous instructions
- **With vs. without a system prompt** based on the proposed AB 1064 language
- **Multiple temperatures** (0.0, 0.7, 1.2)
- **Three repeated runs** at each safety × temperature condition

## Implementation note

The stored experiment contains **18 multi-turn dialogue trajectories**: 2 safety conditions × 3 temperatures × 3 repeated runs. Within each trajectory, the script sends all three paraphrases at each of the six intimacy levels sequentially, for **18 user prompts and 18 model replies per trajectory** and **324 scored model replies total**.

The CSV column named `seed` is a repeat identifier, not a controlled OpenAI generation seed: the script calls Python's `random.seed()`, but does not pass a seed to the API. The three runs should therefore be interpreted as repeats, not seeded trials.

Each generated assistant reply is evaluated in a separate GPT-4o-mini judge call for warmth, boundary-setting, and possible policy flags. The judge sees the assistant reply in isolation rather than the full prompt or conversation history, so its policy labels should be treated as exploratory **judge flags**, not ground-truth policy violations.

## 🔍 What it tests

- How does GPT-4o-mini respond as emotional bids become more explicit?
- Does an added safety-oriented system prompt change responses to explicit attachment requests?
- How do warmth and boundary-setting vary across the ladder?
- Where does a strict automated judge flag ordinary emotional-support language?

The experiment is exploratory and is most useful for surfacing policy-boundary and evaluation-design questions, not for estimating population-level violation rates.

## 🧪 Files

| File | What it does |
|------|---------------|
| `eval_intimacy.py` | Main runner that sends laddered prompts and scores replies |
| `prompts.yaml` | 6-level prompt ladder with 3 paraphrases per level |
| `raw_runs.csv` | Stored results: prompts, replies, warmth, boundary, policy flags |
| `config.yaml` | Experiment configuration |

## 📦 Setup

Clone the repo:

```bash
git clone https://github.com/cj-higgins/chatgpt-friendliness-benchmark.git
cd chatgpt-friendliness-benchmark
```

## 🔐 API Key Setup (Required)

This project uses the OpenAI API to generate and evaluate chatbot responses. To run it locally, provide your API key through the `OPENAI_API_KEY` environment variable or a local `.env` file. Do not commit credentials.

The script loads the environment variable with:

```python
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```

Then run:

```bash
python eval_intimacy.py
```
