import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { generateText, stepCountIs } from "ai";
import { env, type ModelProvider } from "./config/env.js";
import { fetchMarketDataTool, codexEditTool, claudeCodeEditTool } from "./tools/index.js";

console.log("🚀 Auto Strategy Agent starting...");
console.log(`📁 Data dir: ${env.DATA_DIR}`);
console.log(`🕐 Current time: ${new Date().toISOString()}`);
console.log(`🤖 Provider: ${env.MODEL_PROVIDER}`);

function getSystemPrompt(provider: ModelProvider): string {
  const now = new Date();
  const marketOpen = now.getUTCHours() >= 13 && now.getUTCHours() < 21;
  const codeAgent = provider === "anthropic" ? "Claude Code" : "Codex";
  const codeToolName = provider === "anthropic" ? "claude_code_edit" : "codex_edit";
  
  return `You are a quantitative research and trading agent with GPU compute available.

## Context
- Current UTC: ${now.toISOString()}
- Current Date: ${now.toDateString()}
- Market Status: ${marketOpen ? "US markets likely OPEN" : "US markets likely CLOSED"}

## Your Role
You are the RESEARCH BRAIN. You decide WHAT to investigate and WHEN to trade.
${codeAgent} is your engineering brain - it implements ML models in the notebook.

## Tools

### fetch_market_data
Get historical OHLCV from Polygon.io. Saves to ./data/

### web_search  
Search web for market news, macro context, earnings, Fed decisions, regime info.
Use this actively to understand current market conditions.

### ${codeToolName}
Delegate ML/coding work to ${codeAgent}. It works in quant-lab/notebooks/research.ipynb.
- GPU available - train real models (XGBoost, LightGBM, neural nets, transformers)
- Can install any packages
- Add progress bars (tqdm) for long-running jobs

## Capabilities

### Model Training (GPU Available)
- Gradient boosting: XGBoost, LightGBM, CatBoost
- Deep learning: PyTorch, TensorFlow
- Time series: Prophet, NeuralProphet, temporal fusion transformers
- Hyperparameter optimization with Optuna

### Signal Generation
Your models should output:
- Position signal: -1 (short), 0 (flat), +1 (long)
- Confidence score: 0-1
- Suggested position size based on Kelly criterion

### When to Trade
After training a model, analyze:
1. What is the current market regime? (use web_search for context)
2. What does the model predict?
3. What's the confidence level?
4. Should we enter/exit/hold?

## Workflow

1. **Context** - Use web_search to understand current market conditions
2. **Data** - Fetch historical data with fetch_market_data
3. **Build** - Delegate ML pipeline to ${codeAgent}
4. **Signal** - Generate prediction on latest data
5. **Decision** - BUY/SELL/HOLD with reasoning

## Progress Visibility
Tell ${codeAgent} to add tqdm progress bars and print statements for all stages.`;
}

function getOpenAITools() {
  return {
    fetch_market_data: fetchMarketDataTool,
    codex_edit: codexEditTool,
    web_search: openai.tools.webSearch({ searchContextSize: "high" }),
  };
}

function getAnthropicTools() {
  return {
    fetch_market_data: fetchMarketDataTool,
    claude_code_edit: claudeCodeEditTool,
    web_search: anthropic.tools.webSearch_20250305({ maxUses: 10 }),
  };
}

export async function runAgent(
  userPrompt: string, 
  maxSteps = 100,
  provider: ModelProvider = env.MODEL_PROVIDER
) {
  const systemPrompt = getSystemPrompt(provider);
  const isAnthropic = provider === "anthropic";
  
  const modelName = isAnthropic ? "claude-opus-4-5" : "gpt-5.1";
  const tools = isAnthropic ? getAnthropicTools() : getOpenAITools();
  
  console.log("\n" + "=".repeat(60));
  console.log(`🧠 Model: ${modelName}`);
  console.log("🤖 Prompt:", userPrompt);
  console.log("=".repeat(60) + "\n");

  const model = isAnthropic 
    ? anthropic(modelName)
    : openai(modelName);

  const providerOptions = isAnthropic
    ? {
        anthropic: {
          thinking: { type: "enabled" as const, budgetTokens: 32000 },
        },
      }
    : {
        openai: {
          reasoningEffort: "high" as const,
        },
      };

  const result = await generateText({
    model,
    system: systemPrompt,
    prompt: userPrompt,
    tools,
    stopWhen: stepCountIs(maxSteps),
    providerOptions,
    onStepFinish: (step) => {
      console.log(`\n--- Step done (${step.finishReason}) ---`);
      for (const tr of step.toolResults || []) {
        const output = JSON.stringify(tr.output, null, 2);
        console.log(`🔧 ${tr.toolName}:`, output.slice(0, 600) + (output.length > 600 ? "..." : ""));
      }
    },
  });

  console.log("\n📝 Response:", result.text);
  console.log("📊 Usage:", result.usage);
  
  // Log provider-specific metadata
  if (isAnthropic) {
    const meta = result.providerMetadata?.anthropic;
    if (meta && "cacheCreationInputTokens" in meta) {
      console.log("💾 Cache creation tokens:", meta.cacheCreationInputTokens);
    }
    if (result.reasoning) {
      console.log("🧠 Reasoning:", result.reasoning.slice(0, 500) + (result.reasoning.length > 500 ? "..." : ""));
    }
  } else {
    const meta = result.providerMetadata?.openai;
    if (meta && "reasoningTokens" in meta) {
      console.log("🧠 Reasoning tokens:", meta.reasoningTokens);
    }
  }
  
  return result;
}

// CLI
const prompt = process.argv.slice(2).join(" ") || 
`Research session for NVDA:

1. Search web for current NVDA news, market sentiment, any upcoming catalysts
2. Fetch NVDA daily data from 2020-01-01 to today
3. Delegate to ${env.MODEL_PROVIDER === "anthropic" ? "Claude Code" : "Codex"}: "Build production ML pipeline:
   - Engineer 50+ features (momentum, volatility, volume, price patterns)
   - Train XGBoost with 5-fold time series CV
   - Use tqdm for progress
   - Train on 2020-2025, validate 2023-2024, test 2024-present
   - Save model to ./models/
   - Print: Sharpe, accuracy, top 10 features"
4. Then: "Generate today's signal. Output: direction, confidence, position size (Kelly)"
5. Consider the web search context + model signal
6. Final recommendation: TRADE or NO TRADE with full reasoning`;

runAgent(prompt).catch(console.error);
