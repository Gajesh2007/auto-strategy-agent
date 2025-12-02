import { config } from "dotenv";

config();

export type ModelProvider = "openai" | "anthropic";

export const env = {
  OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
  ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY!,
  POLYGON_API_KEY: process.env.POLYGON_API_KEY!,
  DATA_DIR: process.env.DATA_DIR || "./data",
  MODEL_PROVIDER: (process.env.MODEL_PROVIDER || "openai") as ModelProvider,
} as const;

if (!process.env.OPENAI_API_KEY) console.warn("⚠️  Missing OPENAI_API_KEY");
if (!process.env.ANTHROPIC_API_KEY) console.warn("⚠️  Missing ANTHROPIC_API_KEY");
if (!process.env.POLYGON_API_KEY) console.warn("⚠️  Missing POLYGON_API_KEY");
