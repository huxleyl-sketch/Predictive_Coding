import * as tf from '@tensorflow/tfjs';

export type TrainingBatch = {
    x: tf.Tensor2D;
    y: tf.Tensor2D;
};

export type BaselineMetrics = {
    overallMajorityAccuracy: number;
    validationMajorityAccuracy: number;
    validationPersistenceAccuracy: number;
};

export type NiftyTrainingSet = {
    trainBatches: TrainingBatch[];
    trainX: tf.Tensor2D;
    trainY: tf.Tensor2D;
    validationX: tf.Tensor2D;
    validationY: tf.Tensor2D;
    inputSize: number;
    outputSize: number;
    previewX: tf.Tensor2D;
    previewY: tf.Tensor2D;
    sampleCount: number;
    trainCount: number;
    validationCount: number;
    baselines: BaselineMetrics;
};

export type DatasetMode = "raw" | "sma";
export const MARKET_CLASS_LABELS = ["Up", "Flat", "Down"] as const;

type MarketRow = {
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
};

type NormalizationStats = {
    means: number[];
    stds: number[];
};

type WindowSummary = {
    momentum: number;
    meanReturn: number;
    volatility: number;
    meanRange: number;
};

function parseCsvRow(line: string): string[] {
    return line.split(',').map((item) => item.trim());
}

function asNumber(raw: string): number {
    const value = Number(raw);
    return Number.isFinite(value) ? value : NaN;
}

function fitNormalization(rows: number[][]): NormalizationStats {
    if (rows.length === 0) {
        return { means: [], stds: [] };
    }

    const cols = rows[0]!.length;
    const means = new Array(cols).fill(0);
    const stds = new Array(cols).fill(0);

    for (const row of rows) {
        for (let c = 0; c < cols; c++) means[c] += row[c]!;
    }

    for (let c = 0; c < cols; c++) means[c] /= rows.length;

    for (const row of rows) {
        for (let c = 0; c < cols; c++) {
            const d = row[c]! - means[c]!;
            stds[c] += d * d;
        }
    }

    for (let c = 0; c < cols; c++) {
        stds[c] = Math.sqrt(stds[c]! / rows.length);
        if (!Number.isFinite(stds[c]!) || stds[c] === 0) stds[c] = 1;
    }

    return { means, stds };
}

function applyNormalization(rows: number[][], stats: NormalizationStats): number[][] {
    if (rows.length === 0) return rows;

    return rows.map((row) =>
        row.map((value, c) => {
            const mean = stats.means[c] ?? 0;
            const std = stats.stds[c] ?? 1;
            return (value - mean) / std;
        }),
    );
}

function safeRatioChange(next: number, prev: number): number {
    if (!Number.isFinite(next) || !Number.isFinite(prev) || prev === 0) return 0;
    return (next - prev) / Math.abs(prev);
}

function buildDayFeatures(day: MarketRow, previousDay: MarketRow): number[] {
    const openVsPrevClose = safeRatioChange(day.open, previousDay.close);
    const closeVsPrevClose = safeRatioChange(day.close, previousDay.close);
    const intradayReturn = safeRatioChange(day.close, day.open);
    const highVsOpen = safeRatioChange(day.high, day.open);
    const lowVsOpen = safeRatioChange(day.low, day.open);
    const rangeVsOpen = day.open !== 0 ? (day.high - day.low) / Math.abs(day.open) : 0;
    const volumeLog = Math.log1p(Math.max(day.volume, 0));
    const volumeChange = safeRatioChange(day.volume, previousDay.volume);

    return [
        openVsPrevClose,
        closeVsPrevClose,
        intradayReturn,
        highVsOpen,
        lowVsOpen,
        rangeVsOpen,
        volumeLog,
        volumeChange,
    ];
}

function summarizeWindow(rows: MarketRow[], startInclusive: number, endExclusive: number): WindowSummary {
    const returns: number[] = [];
    const ranges: number[] = [];
    for (let i = startInclusive; i < endExclusive; i++) {
        const day = rows[i]!;
        const previousDay = rows[Math.max(0, i - 1)]!;
        returns.push(safeRatioChange(day.close, previousDay.close));
        ranges.push(day.open !== 0 ? (day.high - day.low) / Math.abs(day.open) : 0);
    }

    const firstClose = rows[startInclusive]!.close;
    const lastClose = rows[endExclusive - 1]!.close;
    const momentum = safeRatioChange(lastClose, firstClose);
    const meanReturn = returns.reduce((sum, value) => sum + value, 0) / Math.max(1, returns.length);
    const meanRange = ranges.reduce((sum, value) => sum + value, 0) / Math.max(1, ranges.length);
    let variance = 0;
    for (const value of returns) variance += (value - meanReturn) * (value - meanReturn);
    const volatility = Math.sqrt(variance / Math.max(1, returns.length));

    return { momentum, meanReturn, volatility, meanRange };
}

function toBatches(features: number[][], labels: number[][], batchSize: number): TrainingBatch[] {
    const batches: TrainingBatch[] = [];

    for (let i = 0; i < features.length; i += batchSize) {
        const xRows = features.slice(i, i + batchSize);
        const yRows = labels.slice(i, i + batchSize);

        if (xRows.length === 0 || yRows.length === 0) continue;

        batches.push({
            x: tf.tensor2d(xRows),
            y: tf.tensor2d(yRows),
        });
    }

    return batches;
}

function classFromOneHot(row: number[]): number {
    let bestIndex = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < row.length; i++) {
        const value = row[i] ?? Number.NEGATIVE_INFINITY;
        if (value > bestValue) {
            bestValue = value;
            bestIndex = i;
        }
    }
    return bestIndex;
}

function computeMajorityAccuracy(labels: number[][]): number {
    if (labels.length === 0) return 0;

    let upOrFlat = 0;
    for (const row of labels) {
        if (classFromOneHot(row) === 0) upOrFlat++;
    }

    const down = labels.length - upOrFlat;
    return Math.max(upOrFlat, down) / labels.length;
}

function computePersistenceAccuracy(allLabels: number[][], startIndexInclusive: number): number {
    // Baseline: predict today's class as yesterday's class.
    if (allLabels.length < 2 || startIndexInclusive <= 0 || startIndexInclusive >= allLabels.length) return 0;

    let correct = 0;
    let count = 0;

    for (let i = startIndexInclusive; i < allLabels.length; i++) {
        const predicted = classFromOneHot(allLabels[i - 1]!);
        const actual = classFromOneHot(allLabels[i]!);
        if (predicted === actual) correct++;
        count++;
    }

    return count > 0 ? correct / count : 0;
}

export async function loadNiftyTrainingSet(
    csvPath = `${import.meta.env.BASE_URL}nifty50_last_10_years.csv`,
    batchSize = 128,
    validationRatio = 0.2,
    lookbackDays = 1,
    datasetMode: DatasetMode = "raw",
    smoothingWindow = 5,
    predictionHorizonDays = 3,
    flatThresholdPct = 0.5,
): Promise<NiftyTrainingSet> {
    const response = await fetch(csvPath);
    if (!response.ok) {
        throw new Error(`Failed to fetch dataset at ${csvPath} (${response.status})`);
    }

    const text = await response.text();
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (lines.length < 3) {
        throw new Error('CSV file does not contain enough rows for training.');
    }

    const header = parseCsvRow(lines[0]!);
    const iOpen = header.indexOf('Open');
    const iHigh = header.indexOf('High');
    const iLow = header.indexOf('Low');
    const iClose = header.indexOf('Close');
    const iVolume = header.indexOf('Volume');

    if ([iOpen, iHigh, iLow, iClose, iVolume].some((i) => i < 0)) {
        throw new Error('CSV is missing one or more required columns: Open, High, Low, Close, Volume.');
    }

    const rows: MarketRow[] = [];

    for (let i = 1; i < lines.length; i++) {
        const parts = parseCsvRow(lines[i]!);
        const row: MarketRow = {
            open: asNumber(parts[iOpen] ?? ''),
            high: asNumber(parts[iHigh] ?? ''),
            low: asNumber(parts[iLow] ?? ''),
            close: asNumber(parts[iClose] ?? ''),
            volume: asNumber(parts[iVolume] ?? ''),
        };

        if (
            Number.isFinite(row.open) &&
            Number.isFinite(row.high) &&
            Number.isFinite(row.low) &&
            Number.isFinite(row.close) &&
            Number.isFinite(row.volume)
        ) {
            rows.push(row);
        }
    }

    const activeRows = datasetMode === "sma"
        ? applySimpleMovingAverage(rows, Math.max(2, Math.floor(smoothingWindow)))
        : rows;

    const contextDays = Math.max(1, Math.floor(lookbackDays));
    const horizonDays = Math.max(1, Math.floor(predictionHorizonDays));
    const flatThresholdRatio = Math.max(0, flatThresholdPct) / 100;
    if (activeRows.length < contextDays + horizonDays + 1) {
        throw new Error('Not enough valid market rows after parsing for train/validation split.');
    }

    const featuresRaw: number[][] = [];
    const labels: number[][] = [];

    for (let i = contextDays; i + horizonDays < activeRows.length; i++) {
        const current = activeRows[i]!;
        const future = activeRows[i + horizonDays]!;
        const windowFeatures: number[] = [];

        for (let j = i - contextDays; j < i; j++) {
            const day = activeRows[j]!;
            const previousDay = activeRows[Math.max(0, j - 1)]!;
            windowFeatures.push(...buildDayFeatures(day, previousDay));
        }

        const windowSummary = summarizeWindow(activeRows, i - contextDays, i);
        windowFeatures.push(
            windowSummary.momentum,
            windowSummary.meanReturn,
            windowSummary.volatility,
            windowSummary.meanRange,
        );

        featuresRaw.push(windowFeatures);
        labels.push(labelFromForwardReturn(safeRatioChange(future.close, current.close), flatThresholdRatio));
    }

    const sampleCount = featuresRaw.length;
    const tentativeSplit = Math.floor(sampleCount * (1 - validationRatio));
    const splitIndex = Math.min(sampleCount - 1, Math.max(1, tentativeSplit));

    const trainFeaturesRaw = featuresRaw.slice(0, splitIndex);
    const validationFeaturesRaw = featuresRaw.slice(splitIndex);
    const trainLabels = labels.slice(0, splitIndex);
    const validationLabels = labels.slice(splitIndex);

    const normStats = fitNormalization(trainFeaturesRaw);
    const trainFeatures = applyNormalization(trainFeaturesRaw, normStats);
    const validationFeatures = applyNormalization(validationFeaturesRaw, normStats);

    const trainBatches = toBatches(trainFeatures, trainLabels, Math.max(1, batchSize));
    if (trainBatches.length === 0 || validationFeatures.length === 0) {
        throw new Error('Unable to create non-empty train/validation datasets.');
    }

    const trainX = tf.tensor2d(trainFeatures);
    const trainY = tf.tensor2d(trainLabels);
    const validationX = tf.tensor2d(validationFeatures);
    const validationY = tf.tensor2d(validationLabels);

    const baselines: BaselineMetrics = {
        overallMajorityAccuracy: computeMajorityAccuracy(labels),
        validationMajorityAccuracy: computeMajorityAccuracy(validationLabels),
        validationPersistenceAccuracy: computePersistenceAccuracy(labels, splitIndex),
    };

    return {
        trainBatches,
        trainX,
        trainY,
        validationX,
        validationY,
        inputSize: featuresRaw[0]!.length,
        outputSize: labels[0]!.length,
        previewX: validationX,
        previewY: validationY,
        sampleCount,
        trainCount: trainFeatures.length,
        validationCount: validationFeatures.length,
        baselines,
    };
}

function labelFromForwardReturn(forwardReturn: number, flatThresholdRatio: number): number[] {
    if (forwardReturn > flatThresholdRatio) return [1, 0, 0];
    if (forwardReturn < -flatThresholdRatio) return [0, 0, 1];
    return [0, 1, 0];
}

function applySimpleMovingAverage(rows: MarketRow[], windowSize: number): MarketRow[] {
    if (rows.length === 0) return [];
    const out: MarketRow[] = [];
    for (let i = windowSize - 1; i < rows.length; i++) {
        let open = 0;
        let high = 0;
        let low = 0;
        let close = 0;
        let volume = 0;
        for (let j = i - windowSize + 1; j <= i; j++) {
            const row = rows[j]!;
            open += row.open;
            high += row.high;
            low += row.low;
            close += row.close;
            volume += row.volume;
        }
        const invWindow = 1 / windowSize;
        out.push({
            open: open * invWindow,
            high: high * invWindow,
            low: low * invWindow,
            close: close * invWindow,
            volume: volume * invWindow,
        });
    }
    return out;
}
