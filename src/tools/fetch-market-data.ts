import { z } from "zod";
import { tool } from "ai";
import { writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { env } from "../config/env.js";
import type { OHLCV, MarketDataResult } from "../types/index.js";

const MarketDataSchema = z.object({
  ticker: z.string().describe("Stock ticker symbol, e.g. NVDA, GOOGL, AAPL"),
  startDate: z.string().describe("Start date in YYYY-MM-DD format"),
  endDate: z.string().describe("End date in YYYY-MM-DD format"),
  timespan: z
    .enum(["minute", "hour", "day", "week", "month"])
    .default("day")
    .describe("Timespan for each bar"),
  multiplier: z
    .number()
    .default(1)
    .describe("Multiplier for timespan, e.g. 5 for 5-minute bars"),
});

async function fetchFromPolygon(
  ticker: string,
  startDate: string,
  endDate: string,
  timespan: string,
  multiplier: number
): Promise<OHLCV[]> {
  const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/${multiplier}/${timespan}/${startDate}/${endDate}?adjusted=true&sort=asc&limit=50000&apiKey=${env.POLYGON_API_KEY}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Polygon API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();

  if (!data.results || data.results.length === 0) {
    throw new Error(`No data returned for ${ticker} from ${startDate} to ${endDate}`);
  }

  return data.results.map((bar: Record<string, unknown>) => ({
    timestamp: bar.t as number,
    open: bar.o as number,
    high: bar.h as number,
    low: bar.l as number,
    close: bar.c as number,
    volume: bar.v as number,
    vwap: bar.vw as number | undefined,
    transactions: bar.n as number | undefined,
  }));
}

function saveToFile(ticker: string, timespan: string, data: OHLCV[]): string {
  const dataDir = env.DATA_DIR;
  if (!existsSync(dataDir)) {
    mkdirSync(dataDir, { recursive: true });
  }

  const filename = `${ticker}_${timespan}_${Date.now()}.json`;
  const filePath = join(dataDir, filename);

  writeFileSync(
    filePath,
    JSON.stringify(
      {
        ticker,
        timespan,
        fetchedAt: new Date().toISOString(),
        dataPoints: data.length,
        data,
      },
      null,
      2
    )
  );

  return filePath;
}

export const fetchMarketDataTool = tool({
  description:
    "Fetch historical market data (OHLCV) for a stock ticker from Polygon.io and save it to disk. Returns file path and summary statistics.",
  inputSchema: MarketDataSchema,
  execute: async ({ ticker, startDate, endDate, timespan, multiplier }): Promise<MarketDataResult> => {
    console.log(`📊 Fetching ${ticker} data from ${startDate} to ${endDate}...`);

    const data = await fetchFromPolygon(ticker, startDate, endDate, timespan, multiplier);

    const filePath = saveToFile(ticker, timespan, data);

    const closes = data.map((d) => d.close);
    const volumes = data.map((d) => d.volume);

    const firstTimestamp = data[0]?.timestamp;
    const lastTimestamp = data[data.length - 1]?.timestamp;

    const result: MarketDataResult = {
      ticker,
      timespan,
      startDate,
      endDate,
      dataPoints: data.length,
      filePath,
      summary: {
        firstDate: firstTimestamp ? new Date(firstTimestamp).toISOString().split("T")[0]! : startDate,
        lastDate: lastTimestamp ? new Date(lastTimestamp).toISOString().split("T")[0]! : endDate,
        minPrice: Math.min(...closes),
        maxPrice: Math.max(...closes),
        avgVolume: volumes.reduce((a, b) => a + b, 0) / volumes.length,
      },
    };

    console.log(`✅ Saved ${data.length} bars to ${filePath}`);
    return result;
  },
});
