import { z } from "zod";
import { tool } from "ai";
import { Codex } from "@openai/codex-sdk";
import type { CodexEditResult } from "../types/index.js";
import { resolve } from "path";
import { env } from "../config/env.js";

const CodexEditSchema = z.object({
  taskDescription: z
    .string()
    .describe("Detailed ML/coding task. Be specific: model architecture, features, training params, output format."),
});

let codexInstance: Codex | null = null;

function getCodex(): Codex {
  if (!codexInstance) {
    // SDK sets CODEX_API_KEY from apiKey option for the CLI subprocess
    codexInstance = new Codex({
      apiKey: env.OPENAI_API_KEY,
    });
  }
  return codexInstance;
}

const CODEX_SYSTEM_CONTEXT = `You are a quant ML engineer with GPU compute available.

WORKSPACE: quant-lab/
├── notebooks/research.ipynb  <-- ALL CODE GOES HERE
└── backtest/                 <-- Helper modules
    ├── features.py           <-- 100+ technical indicators
    ├── models.py             <-- XGB/LGB/RF wrappers
    └── runner.py             <-- Backtest engine

ENVIRONMENT:
- GPU available for training
- Can install any package: !pip install <pkg> or !poetry add <pkg>
- Heavy compute OK: hyperparameter sweeps, neural nets, transformers

RULES:
1. ALL code in notebooks/research.ipynb - add new cells, don't delete working code
2. ALWAYS add progress visibility:
   - from tqdm import tqdm for loops
   - print() statements for stages
   - Checkpointing for long jobs
3. Document with markdown cells
4. Save trained models to ./models/ directory

AVAILABLE IMPORTS:
- pandas, numpy, matplotlib, plotly
- sklearn, xgboost, lightgbm, catboost
- torch, tensorflow (install if needed)
- optuna for hyperparameter optimization
- from backtest.runner import run_backtest, load_market_data
- from backtest.features import add_all_features, get_feature_list
- from backtest.models import get_model, train_model, predict_signals

OUTPUT FORMAT FOR SIGNALS:
When generating trading signals, output:
{
  "timestamp": "2024-01-15",
  "ticker": "NVDA",
  "signal": 1,  # -1=short, 0=flat, 1=long
  "confidence": 0.73,
  "predicted_return": 0.012,
  "position_size": 0.25,  # Kelly or risk-adjusted
  "reasoning": "Strong momentum, low vol regime"
}

PROGRESS TEMPLATE:
\`\`\`python
from tqdm import tqdm
import joblib
from pathlib import Path

Path("./models").mkdir(exist_ok=True)

print("Stage 1: Loading data...")
# ... code ...

print("Stage 2: Feature engineering...")
# ... code ...

print("Stage 3: Training model...")
for epoch in tqdm(range(n_epochs), desc="Training"):
    # ... training code ...

print("Stage 4: Evaluation...")
# ... evaluation ...

print("Stage 5: Saving model...")
joblib.dump(model, "./models/model_v1.pkl")

print("✅ Complete!")
\`\`\``;

export const codexEditTool = tool({
  description: `Delegate ML/coding to Codex. GPU available for heavy compute.

Examples:
- "Train XGBoost with 5-fold CV, show progress with tqdm"
- "Build PyTorch LSTM for 60-day sequences, train 100 epochs"
- "Run Optuna optimization, 200 trials"
- "Generate today's signal from trained model"`,
  inputSchema: CodexEditSchema,
  execute: async ({ taskDescription }): Promise<CodexEditResult> => {
    const workingDirectory = resolve(process.cwd(), "quant-lab");
    
    console.log(`\n🤖 Codex starting...`);
    console.log(`📝 ${taskDescription.slice(0, 300)}${taskDescription.length > 300 ? "..." : ""}`);

    try {
      const codex = getCodex();
      const thread = codex.startThread({
        workingDirectory,
        skipGitRepoCheck: true,
        sandboxMode: "workspace-write",  // Allow file modifications
      });

      const fullPrompt = `${CODEX_SYSTEM_CONTEXT}

---
TASK:
${taskDescription}

Remember: Add to notebooks/research.ipynb. Use tqdm for progress. Print stage updates.`;

      const turn = await thread.run(fullPrompt);

      const logs: string[] = [];
      const changedFiles: string[] = [];

      for (const item of turn.items) {
        if (item.type === "agent_message" && "content" in item) {
          logs.push(String(item.content));
        }
        if (item.type === "file_change" && "path" in item) {
          changedFiles.push(String(item.path));
        }
        if (item.type === "command_execution" && "command" in item) {
          logs.push(`$ ${item.command}`);
        }
      }

      console.log(`✅ Codex done. Files: ${changedFiles.length > 0 ? changedFiles.join(", ") : "none"}`);
      
      return {
        summary: turn.finalResponse || "Completed",
        changedFiles,
        logs,
        success: true,
      };
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error(`❌ Codex error: ${msg}`);
      return { summary: `Failed: ${msg}`, changedFiles: [], logs: [msg], success: false };
    }
  },
});
