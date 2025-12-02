import { z } from "zod";
import { tool } from "ai";
import { query } from "@anthropic-ai/claude-agent-sdk";
import type { CodexEditResult } from "../types/index.js";
import { resolve } from "path";

const ClaudeCodeEditSchema = z.object({
  taskDescription: z
    .string()
    .describe("Detailed ML/coding task. Be specific: model architecture, features, training params, output format."),
});

const CLAUDE_CODE_SYSTEM = `You are a quant ML engineer with GPU compute available.

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

export const claudeCodeEditTool = tool({
  description: `Delegate ML/coding to Claude Code. GPU available for heavy compute.

Examples:
- "Train XGBoost with 5-fold CV, show progress with tqdm"
- "Build PyTorch LSTM for 60-day sequences, train 100 epochs"
- "Run Optuna optimization, 200 trials"
- "Generate today's signal from trained model"`,
  inputSchema: ClaudeCodeEditSchema,
  execute: async ({ taskDescription }): Promise<CodexEditResult> => {
    const workingDirectory = resolve(process.cwd(), "quant-lab");
    
    console.log(`\n🟣 Claude Code starting...`);
    console.log(`📝 ${taskDescription.slice(0, 300)}${taskDescription.length > 300 ? "..." : ""}`);

    try {
      const fullPrompt = `${taskDescription}

Remember: Add to notebooks/research.ipynb. Use tqdm for progress. Print stage updates.`;

      const logs: string[] = [];
      const changedFiles: string[] = [];
      let finalResult = "";
      let sessionId = "";

      const result = query({
        prompt: fullPrompt,
        options: {
          cwd: workingDirectory,
          systemPrompt: CLAUDE_CODE_SYSTEM,
          permissionMode: "bypassPermissions",
          model: "claude-opus-4-5",
          allowedTools: [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "NotebookEdit",
          ],
        },
      });

      for await (const message of result) {
        if (message.type === "system" && message.subtype === "init") {
          sessionId = message.session_id;
          console.log(`🔗 Session: ${sessionId}`);
        }

        if (message.type === "assistant") {
          const content = message.message.content;
          for (const block of content) {
            if (block.type === "text") {
              logs.push(block.text);
            }
            if (block.type === "tool_use") {
              logs.push(`🔧 Tool: ${block.name}`);
              
              // Track file changes
              if (["Write", "Edit", "NotebookEdit"].includes(block.name)) {
                const input = block.input as Record<string, unknown>;
                const filePath = (input.file_path || input.notebook_path || input.path) as string | undefined;
                if (filePath && !changedFiles.includes(filePath)) {
                  changedFiles.push(filePath);
                }
              }
            }
          }
        }

        if (message.type === "result") {
          if (message.subtype === "success") {
            finalResult = message.result;
            console.log(`✅ Claude Code done. Cost: $${message.total_cost_usd.toFixed(4)}`);
            console.log(`📊 Turns: ${message.num_turns}, Duration: ${(message.duration_ms / 1000).toFixed(1)}s`);
          } else {
            console.log(`⚠️  Claude Code finished with: ${message.subtype}`);
          }
        }
      }

      console.log(`📁 Files: ${changedFiles.length > 0 ? changedFiles.join(", ") : "none"}`);
      
      return {
        summary: finalResult || "Completed",
        changedFiles,
        logs,
        success: true,
      };
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error(`❌ Claude Code error: ${msg}`);
      return { summary: `Failed: ${msg}`, changedFiles: [], logs: [msg], success: false };
    }
  },
});

