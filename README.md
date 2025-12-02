# Auto Strategy Agent

Self-iterating quant agent with GPU compute. Trains real ML models, generates trading signals.

## Architecture

```
┌────────────────────────────────────────────┐
│         ORCHESTRATOR (GPT-4o)              │
│     Research brain - decides WHAT          │
│   fetch_market_data | web_search           │
└──────────────────┬─────────────────────────┘
                   │ codex_edit
                   ▼
┌────────────────────────────────────────────┐
│           CODEX (GPU Available)            │
│      Engineering brain - builds HOW        │
│   quant-lab/notebooks/research.ipynb       │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│             ML OUTPUTS                     │
│   models/*.pkl | signals | decisions      │
└────────────────────────────────────────────┘
```

## Setup

```bash
pnpm install
cd quant-lab && poetry install --no-root

cp .env.example .env
# Add OPENAI_API_KEY, POLYGON_API_KEY

pnpm run agent "your prompt"
```

## Capabilities

### Model Training (GPU)
- XGBoost, LightGBM, CatBoost
- PyTorch, TensorFlow (install as needed)
- Optuna hyperparameter optimization
- Progress bars via tqdm

### Signal Generation
Models output:
```json
{
  "signal": 1,
  "confidence": 0.73,
  "position_size": 0.25,
  "reasoning": "Strong momentum"
}
```

## Example

```bash
pnpm run agent "
  Fetch NVDA 2020-today.
  Train XGBoost with 50+ features, 5-fold CV.
  Generate today's signal with confidence.
  Recommend: TRADE or NO TRADE.
"
```

## Structure

```
auto-strategy-agent/
├── src/
│   ├── index.ts           # Orchestrator
│   └── tools/
│       ├── fetch-market-data.ts
│       └── codex-agent.ts
├── quant-lab/
│   ├── notebooks/research.ipynb
│   ├── models/            # Saved models
│   └── backtest/          # Python modules
└── data/                  # Market data
```
