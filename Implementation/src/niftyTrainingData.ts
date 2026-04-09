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
    return row[0]! >= row[1]! ? 0 : 1;
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
    csvPath = '/nifty50_last_10_years.csv',
    batchSize = 128,
    validationRatio = 0.2,
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

    if (rows.length < 20) {
        throw new Error('Not enough valid market rows after parsing for train/validation split.');
    }

    const featuresRaw: number[][] = [];
    const labels: number[][] = [];

    for (let i = 1; i < rows.length; i++) {
        const prev = rows[i - 1]!;
        const current = rows[i]!;

        featuresRaw.push([
            prev.open,
            prev.high,
            prev.low,
            prev.close,
            Math.log1p(Math.max(prev.volume, 0)),
        ]);

        labels.push(current.close >= prev.close ? [1, 0] : [0, 1]);
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
