import { config } from "dotenv";

config();

export const env = {
  OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
  POLYGON_API_KEY: process.env.POLYGON_API_KEY!,
  DATA_DIR: process.env.DATA_DIR || "./data",
} as const;

if (!process.env.OPENAI_API_KEY) console.warn("⚠️  Missing OPENAI_API_KEY");
if (!process.env.POLYGON_API_KEY) console.warn("⚠️  Missing POLYGON_API_KEY");
