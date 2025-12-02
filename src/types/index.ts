export interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
  transactions?: number;
}

export interface MarketDataResult {
  ticker: string;
  timespan: string;
  startDate: string;
  endDate: string;
  dataPoints: number;
  filePath: string;
  summary: {
    firstDate: string;
    lastDate: string;
    minPrice: number;
    maxPrice: number;
    avgVolume: number;
  };
}

export interface CodexEditResult {
  summary: string;
  changedFiles: string[];
  logs: string[];
  success: boolean;
}
