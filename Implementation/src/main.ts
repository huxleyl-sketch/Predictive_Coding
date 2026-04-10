import * as tf from '@tensorflow/tfjs';
import { loadNiftyTrainingSet, type TrainingBatch } from './niftyTrainingData';
import { PCN, type PCNConfig } from './pcn';
import { ErrorGraph, InterpretationGraph, SuccessGraph } from './successGraph';
import './style.css';

// Ensure a backend is registered before any eager ops run.
import '@tensorflow/tfjs-backend-webgl';
await tf.setBackend('webgl');
await tf.ready();

const debug = document.getElementById("debug");
const trainButton = document.getElementById("train") as HTMLInputElement | null;
const haltButton = document.getElementById("halt") as HTMLInputElement | null;
const generateButton = document.getElementById("retry") as HTMLInputElement | null;
const cfgDebug = document.getElementById("cfgDebug") as HTMLInputElement | null;
const cfgSignedOutput = document.getElementById("cfgSignedOutput") as HTMLInputElement | null;
const cfgHaltNonFinite = document.getElementById("cfgHaltNonFinite") as HTMLInputElement | null;
const cfgBatchSize = document.getElementById("cfgBatchSize") as HTMLInputElement | null;
const cfgValidationRatio = document.getElementById("cfgValidationRatio") as HTMLInputElement | null;
const cfgEpochs = document.getElementById("cfgEpochs") as HTMLInputElement | null;
const cfgInferSteps = document.getElementById("cfgInferSteps") as HTMLInputElement | null;
const cfgLearnSteps = document.getElementById("cfgLearnSteps") as HTMLInputElement | null;
const cfgInferLr = document.getElementById("cfgInferLr") as HTMLInputElement | null;
const cfgLearnLr = document.getElementById("cfgLearnLr") as HTMLInputElement | null;
const cfgGradClip = document.getElementById("cfgGradClip") as HTMLInputElement | null;
const cfgHiddenLayers = document.getElementById("cfgHiddenLayers") as HTMLInputElement | null;
const cfgHidden1 = document.getElementById("cfgHidden1") as HTMLInputElement | null;
const cfgHidden2 = document.getElementById("cfgHidden2") as HTMLInputElement | null;
const cfgHidden3 = document.getElementById("cfgHidden3") as HTMLInputElement | null;
const cfgHidden4 = document.getElementById("cfgHidden4") as HTMLInputElement | null;
const cfgBalanceFalseError = document.getElementById("cfgBalanceFalseError") as HTMLInputElement | null;
const cfgFalseBalanceStrength = document.getElementById("cfgFalseBalanceStrength") as HTMLInputElement | null;
const trainSuccessGraph = new SuccessGraph(
    document.getElementById("successCanvas") as HTMLCanvasElement | null,
    { title: "Train Accuracy", xAxisLabel: "Epoch" },
);
const validationSuccessGraph = new SuccessGraph(
    document.getElementById("validationCanvas") as HTMLCanvasElement | null,
    { title: "Validation Accuracy", xAxisLabel: "Epoch/Run" },
);
const errorGraph = new ErrorGraph(
    document.getElementById("errorCanvas") as HTMLCanvasElement | null,
);
const interpretationGraph = new InterpretationGraph(
    document.getElementById("interpretationCanvas") as HTMLCanvasElement | null,
);
trainSuccessGraph.render();
validationSuccessGraph.render();
errorGraph.render();
interpretationGraph.render();

let upPcn: PCN | null = null;
let downPcn: PCN | null = null;
let previewX: tf.Tensor2D | null = null;
let previewY: tf.Tensor2D | null = null;
let isTraining = false;
let haltRequested = false;
const TRAIN_HALTED_ERROR = "TRAIN_HALTED_BY_USER";

type TrainingUiConfig = {
    useSignedOutputMapping: boolean;
    haltOnNonFinite: boolean;
    pcnDebug: boolean;
    batchSize: number;
    validationRatio: number;
    epochs: number;
    T_infer: number;
    T_learn: number;
    eta_infer: number;
    eta_learn: number;
    gradientClipValue: number;
    hiddenLayerCount: number;
    hiddenSizes: number[];
    balanceFalseError: boolean;
    falseErrorBalanceStrength: number;
};

if (debug) debug.innerText = "Ready. Click Train.";
if (haltButton) haltButton.disabled = true;
updateHiddenConfigVisibility();
cfgHiddenLayers?.addEventListener("input", () => updateHiddenConfigVisibility());

async function trainFromNiftyCsv() {
    logTfMemory("Before training load");
    const ui = readTrainingConfig();
    const dataset = await loadNiftyTrainingSet('/nifty50_last_10_years.csv', ui.batchSize, ui.validationRatio);
    let downLabelsTrain: tf.Tensor2D | null = null;
    let downLabelsValidation: tf.Tensor2D | null = null;
    let downTrainBatches: TrainingBatch[] = [];
    let nextUpPcn: PCN | null = null;
    let nextDownPcn: PCN | null = null;
    let trainingCompleted = false;

    try {
        // ------------------------------------------------------------------------
        // Dual-PCN setup:
        // 1) upPcn   learns "Up/Flat" vs "not Up/Flat"
        // 2) downPcn learns "Down"    vs "not Down"
        //
        // We keep 2 output neurons for each PCN:
        //   [positive_class, negative_class]
        // This keeps compatibility with existing argmax-based accuracy/visualization.
        // ------------------------------------------------------------------------

        const upLabelsTrain = dataset.trainY;
        const upLabelsValidation = dataset.validationY;

        // For the DOWN model we swap the two columns:
        //   original [Up, Down] -> [Down, Up]
        // so column 0 is now the "positive" Down class for this model.
        downLabelsTrain = swapBinaryOneHot(upLabelsTrain);
        downLabelsValidation = swapBinaryOneHot(upLabelsValidation);
        downTrainBatches = createSwappedLabelBatches(dataset.trainBatches);

        const commonSetup: Omit<PCNConfig, 'data' | 'trainEvalData' | 'validationEvalData'> = {
            epochs: ui.epochs,
            T_infer: ui.T_infer,
            eta_infer: ui.eta_infer,
            T_learn: ui.T_learn,
            eta_learn: ui.eta_learn,
            useSignedOutputMapping: ui.useSignedOutputMapping,
            gradientClipValue: ui.gradientClipValue,
            balanceFalseError: ui.balanceFalseError,
            falseErrorBalanceStrength: ui.falseErrorBalanceStrength,
            haltOnNonFinite: ui.haltOnNonFinite,
            debug: ui.pcnDebug,
            collectAccuracyHistory: true,
            evalInferSteps: Math.max(10, ui.T_infer),
        };

        const upSetup: PCNConfig = {
            ...commonSetup,
            stopRequested: () => haltRequested,
            trainEvalData: { x: dataset.trainX, y: upLabelsTrain },
            validationEvalData: { x: dataset.validationX, y: upLabelsValidation },
            data: dataset.trainBatches,
        };

        const downSetup: PCNConfig = {
            ...commonSetup,
            stopRequested: () => haltRequested,
            trainEvalData: { x: dataset.trainX, y: downLabelsTrain },
            validationEvalData: { x: dataset.validationX, y: downLabelsValidation },
            data: downTrainBatches,
        };

        const dims = [
            dataset.inputSize,
            ...buildHiddenSizes(ui.hiddenLayerCount, ui.hiddenSizes),
            dataset.outputSize,
        ];
        nextUpPcn = new PCN(dims);
        nextDownPcn = new PCN(dims);

        console.log(
            `Loaded ${dataset.sampleCount} samples from NIFTY50 CSV ` +
            `(train=${dataset.trainCount}, validation=${dataset.validationCount}).`,
        );
        console.log(
            `Validation baselines -> majority=${(dataset.baselines.validationMajorityAccuracy * 100).toFixed(2)}%, ` +
            `persistence=${(dataset.baselines.validationPersistenceAccuracy * 100).toFixed(2)}%`,
        );
        console.log(`PCN architecture -> ${dims.join(" -> ")}`);

        // Train both models.
        const upReport = await nextUpPcn.Train(upSetup);
        if (haltRequested) throw new Error(TRAIN_HALTED_ERROR);
        const downReport = await nextDownPcn.Train(downSetup);
        if (haltRequested) throw new Error(TRAIN_HALTED_ERROR);

        // Keep previous model alive until we have a new successfully trained one.
        upPcn?.dispose();
        downPcn?.dispose();
        upPcn = nextUpPcn;
        downPcn = nextDownPcn;
        trainingCompleted = true;

        // Replace preview tensors with cloned validation tensors from this run.
        previewX?.dispose();
        previewY?.dispose();
        previewX = dataset.validationX.clone();
        previewY = dataset.validationY.clone();

        // Graphs: show average of both models' training diagnostics.
        errorGraph.setHistory(averageHistories(upReport.epochMSE, downReport.epochMSE));
        trainSuccessGraph.setHistory(averageHistories(upReport.epochTrainAccuracy, downReport.epochTrainAccuracy));
        validationSuccessGraph.setHistory(
            averageHistories(upReport.epochValidationAccuracy, downReport.epochValidationAccuracy),
        );
        validationSuccessGraph.setBaselines([
            { label: "Majority", value: dataset.baselines.validationMajorityAccuracy, color: "#f59e0b" },
            { label: "Persistence", value: dataset.baselines.validationPersistenceAccuracy, color: "#6366f1" },
        ]);

        // Post-train validation pass using *combined* decision from both PCNs.
        const upPred = nextUpPcn.GenerateMapped(previewX, 50, 0.05);
        const downPred = nextDownPcn.GenerateMapped(previewX, 50, 0.05);

        const combinedPredClass = combineDualPcnVotes(upPred, downPred);
        const upPredClass = classIds(upPred);
        const downPredClass = classIds(downPred);
        const upTargetClass = classIds(previewY);
        const downTargetClass = classIdsForDownModel(previewY);

        interpretationGraph.setDualPcnBinary(
            upPredClass,
            upTargetClass,
            downPredClass,
            downTargetClass,
            { upPositiveClassName: "Up/Flat", downPositiveClassName: "Down" },
        );

        const postTrainSuccess = classAccuracy(combinedPredClass, upTargetClass);
        validationSuccessGraph.record(postTrainSuccess);

        upPred.dispose();
        downPred.dispose();

        if (debug) {
            debug.innerText =
                `Post-train validation: ${(postTrainSuccess * 100).toFixed(2)}% ` +
                `(majority ${(dataset.baselines.validationMajorityAccuracy * 100).toFixed(2)}%)`;
        }
    } finally {
        // Dispose temporary tensors created for down-model label swapping.
        downLabelsTrain?.dispose();
        downLabelsValidation?.dispose();
        for (const batch of downTrainBatches) {
            batch.y.dispose();
        }

        // Dispose dataset tensors; preview tensors are cloned above.
        for (const batch of dataset.trainBatches) {
            batch.x.dispose();
            batch.y.dispose();
        }
        dataset.trainX.dispose();
        dataset.trainY.dispose();
        dataset.validationX.dispose();
        dataset.validationY.dispose();

        // If this run failed/halted before replacing active models, release new models.
        if (!trainingCompleted) {
            nextUpPcn?.dispose();
            nextDownPcn?.dispose();
        }
        logTfMemory("After training cleanup");
    }
}

try {
    trainButton?.addEventListener('click', async () => {
        if (isTraining) return;
        isTraining = true;
        haltRequested = false;
        if (trainButton) trainButton.disabled = true;
        if (haltButton) haltButton.disabled = false;
        if (generateButton) generateButton.disabled = true;
        if (debug) debug.innerText = "Training...";

        try {
            await trainFromNiftyCsv();
        } catch (error) {
            const haltedByUser = error instanceof Error && error.message === TRAIN_HALTED_ERROR;
            if (haltedByUser) {
                if (debug) debug.innerText = "Training halted.";
            } else {
                console.error('Failed to load/train from NIFTY50 CSV:', error);
                if (debug) debug.innerText = "Training failed (check console)";
            }
        } finally {
            isTraining = false;
            haltRequested = false;
            if (trainButton) trainButton.disabled = false;
            if (haltButton) haltButton.disabled = true;
            if (generateButton) generateButton.disabled = false;
        }
    });
} catch (error) {
    console.error('Failed to initialize controls:', error);
    if (debug) debug.innerText = "UI init failed (check console)";
}

haltButton?.addEventListener('click', () => {
    if (!isTraining) return;
    haltRequested = true;
    haltButton.disabled = true;
    if (debug) debug.innerText = "Halting training after current batch...";
});

generateButton?.addEventListener('click', () => {
    if (!upPcn || !downPcn || !previewX || !previewY) {
        console.warn("Model is not trained yet. Click Train first.");
        if (debug) debug.innerText = "Train the model first.";
        return;
    }

    // Run both PCNs and combine their votes into one 2-class prediction.
    const upPred = upPcn.GenerateMapped(previewX, 50, 0.05);
    const downPred = downPcn.GenerateMapped(previewX, 50, 0.05);

    const combinedPredClass = combineDualPcnVotes(upPred, downPred);
    const upPredClass = classIds(upPred);
    const downPredClass = classIds(downPred);
    const upTargetClass = classIds(previewY);
    const downTargetClass = classIdsForDownModel(previewY);

    interpretationGraph.setDualPcnBinary(
        upPredClass,
        upTargetClass,
        downPredClass,
        downTargetClass,
        { upPositiveClassName: "Up/Flat", downPositiveClassName: "Down" },
    );
    const success = validationSuccessGraph.record(classAccuracy(combinedPredClass, upTargetClass));

    upPred.dispose();
    downPred.dispose();
    logTfMemory("After generate");

    if (debug) debug.innerText = `Generate success: ${(success * 100).toFixed(2)}%`;
});

function classIds(mat: tf.Tensor2D): Int32Array {
    const classes = tf.argMax(mat, 1);
    const ids = Int32Array.from(classes.dataSync());
    classes.dispose();
    return ids;
}

function readTrainingConfig(): TrainingUiConfig {
    return {
        useSignedOutputMapping: checkboxOrDefault(cfgSignedOutput, false),
        haltOnNonFinite: checkboxOrDefault(cfgHaltNonFinite, true),
        pcnDebug: checkboxOrDefault(cfgDebug, false),
        batchSize: intOrDefault(cfgBatchSize, 128, 1),
        validationRatio: numberOrDefault(cfgValidationRatio, 0.2, 0.05, 0.45),
        epochs: intOrDefault(cfgEpochs, 40, 1),
        T_infer: intOrDefault(cfgInferSteps, 25, 1),
        T_learn: intOrDefault(cfgLearnSteps, 25, 1),
        eta_infer: numberOrDefault(cfgInferLr, 0.01, 1e-6),
        eta_learn: numberOrDefault(cfgLearnLr, 0.0001, 1e-7),
        gradientClipValue: numberOrDefault(cfgGradClip, 3, 0.01),
        hiddenLayerCount: intOrDefault(cfgHiddenLayers, 2, 1, 4),
        hiddenSizes: [
            intOrDefault(cfgHidden1, 667, 1),
            intOrDefault(cfgHidden2, 333, 1),
            intOrDefault(cfgHidden3, 167, 1),
            intOrDefault(cfgHidden4, 83, 1),
        ],
        balanceFalseError: checkboxOrDefault(cfgBalanceFalseError, false),
        falseErrorBalanceStrength: numberOrDefault(cfgFalseBalanceStrength, 1.5, 0),
    };
}

function buildHiddenSizes(hiddenLayerCount: number, hiddenSizes: number[]): number[] {
    return hiddenSizes.slice(0, hiddenLayerCount).map((v) => Math.max(1, Math.floor(v)));
}

function updateHiddenConfigVisibility() {
    const count = intOrDefault(cfgHiddenLayers, 2, 1, 4);
    const rows = document.querySelectorAll<HTMLElement>("[data-hidden-index]");
    for (const row of rows) {
        const idx = Number(row.dataset.hiddenIndex ?? "0");
        row.style.display = idx >= 1 && idx <= count ? "" : "none";
    }
}

function checkboxOrDefault(input: HTMLInputElement | null, fallback: boolean): boolean {
    return input ? input.checked : fallback;
}

function intOrDefault(
    input: HTMLInputElement | null,
    fallback: number,
    min: number,
    max?: number,
): number {
    const raw = Number(input?.value ?? fallback);
    const finite = Number.isFinite(raw) ? Math.floor(raw) : fallback;
    const clampedMin = Math.max(min, finite);
    return max === undefined ? clampedMin : Math.min(max, clampedMin);
}

function numberOrDefault(
    input: HTMLInputElement | null,
    fallback: number,
    min?: number,
    max?: number,
): number {
    const raw = Number(input?.value ?? fallback);
    let value = Number.isFinite(raw) ? raw : fallback;
    if (min !== undefined) value = Math.max(min, value);
    if (max !== undefined) value = Math.min(max, value);
    return value;
}

function logTfMemory(label: string) {
    const m = tf.memory();
    console.log(
        `${label} | tensors=${m.numTensors}, bytes=${Math.round(m.numBytes / (1024 * 1024) * 100) / 100} MB`,
    );
}

function classIdsForDownModel(upDownLabels: tf.Tensor2D): Int32Array {
    const swapped = swapBinaryOneHot(upDownLabels);
    const ids = classIds(swapped);
    swapped.dispose();
    return ids;
}

function classAccuracy(predicted: ArrayLike<number>, target: ArrayLike<number>): number {
    const n = Math.min(predicted.length, target.length);
    if (n === 0) return 0;

    let correct = 0;
    for (let i = 0; i < n; i++) {
        if (Number(predicted[i]) === Number(target[i])) correct++;
    }
    return correct / n;
}

function averageHistories(a: number[], b: number[]): number[] {
    const n = Math.min(a.length, b.length);
    if (n === 0) return [];
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) out[i] = (a[i]! + b[i]!) / 2;
    return out;
}

function swapBinaryOneHot(y: tf.Tensor2D): tf.Tensor2D {
    // [col0, col1] -> [col1, col0]
    const col1 = y.slice([0, 1], [-1, 1]);
    const col0 = y.slice([0, 0], [-1, 1]);
    const swapped = tf.concat([col1, col0], 1) as tf.Tensor2D;
    col1.dispose();
    col0.dispose();
    return swapped;
}

function createSwappedLabelBatches(batches: TrainingBatch[]): TrainingBatch[] {
    return batches.map((batch) => ({
        x: batch.x,
        y: swapBinaryOneHot(batch.y),
    }));
}

function combineDualPcnVotes(upPred: tf.Tensor2D, downPred: tf.Tensor2D): Int32Array {
    // Both models output [positive, negative] for each sample.
    // We convert each output to a margin score:
    //   margin = positive - negative
    // Then choose final class from the stronger positive margin:
    //   if upMargin >= downMargin => class 0 (Up/Flat)
    //   else                     => class 1 (Down)
    const up = upPred.dataSync();
    const down = downPred.dataSync();

    const n = Math.min(upPred.shape[0], downPred.shape[0]);
    const out = new Int32Array(n);

    for (let i = 0; i < n; i++) {
        const base = i * 2;
        const upMargin = up[base]! - up[base + 1]!;
        const downMargin = down[base]! - down[base + 1]!;
        out[i] = upMargin >= downMargin ? 0 : 1;
    }

    return out;
}
