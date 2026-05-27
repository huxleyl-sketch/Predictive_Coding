import * as tf from '@tensorflow/tfjs';
import {
    MARKET_CLASS_LABELS,
    loadNiftyTrainingSet,
    type DatasetMode,
    type NiftyTrainingSet,
} from './niftyTrainingData';
import { PCN, type PCNConfig } from './pcn';
import { StandardClassifier } from './standardModel';
import { ErrorGraph, InterpretationGraph, SuccessGraph } from './successGraph';
import './style.css';

import '@tensorflow/tfjs-backend-webgl';
await tf.setBackend('webgl');
await tf.ready();

type MarketKey = "nifty" | "sp500";
type ModelKind = "pcn" | "standard";
type TrainableModel = PCN | StandardClassifier;

type TrainingUiConfig = {
    modelKind: ModelKind;
    datasetMode: DatasetMode;
    smoothingWindow: number;
    predictionHorizonDays: number;
    flatThresholdPct: number;
    useSignedOutputMapping: boolean;
    haltOnNonFinite: boolean;
    pcnDebug: boolean;
    batchSize: number;
    lookbackDays: number;
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

type MarketState = {
    label: string;
    csvPath: string;
    debug: HTMLElement | null;
    crossDebug: HTMLElement | null;
    trainButton: HTMLInputElement | null;
    haltButton: HTMLInputElement | null;
    generateButton: HTMLInputElement | null;
    crossTestButton: HTMLInputElement | null;
    trainSuccessGraph: SuccessGraph;
    validationSuccessGraph: SuccessGraph;
    errorGraph: ErrorGraph;
    interpretationGraph: InterpretationGraph;
    model: TrainableModel | null;
    previewX: tf.Tensor2D | null;
    previewY: tf.Tensor2D | null;
    lastTrainingConfig: TrainingUiConfig | null;
    isTraining: boolean;
    haltRequested: boolean;
};

const TRAIN_HALTED_ERROR = "TRAIN_HALTED_BY_USER";
const publicAsset = (name: string): string => `${import.meta.env.BASE_URL}${name}`;

const cfgDebug = document.getElementById("cfgDebug") as HTMLInputElement | null;
const cfgSignedOutput = document.getElementById("cfgSignedOutput") as HTMLInputElement | null;
const cfgHaltNonFinite = document.getElementById("cfgHaltNonFinite") as HTMLInputElement | null;
const cfgBatchSize = document.getElementById("cfgBatchSize") as HTMLInputElement | null;
const cfgModelKind = document.getElementById("cfgModelKind") as HTMLSelectElement | null;
const cfgLookbackDays = document.getElementById("cfgLookbackDays") as HTMLInputElement | null;
const cfgDatasetMode = document.getElementById("cfgDatasetMode") as HTMLSelectElement | null;
const cfgSmoothingWindow = document.getElementById("cfgSmoothingWindow") as HTMLInputElement | null;
const cfgPredictionHorizonDays = document.getElementById("cfgPredictionHorizonDays") as HTMLInputElement | null;
const cfgFlatThresholdPct = document.getElementById("cfgFlatThresholdPct") as HTMLInputElement | null;
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

const marketStates: Record<MarketKey, MarketState> = {
    nifty: createMarketState("nifty", "NIFTY 50", publicAsset("nifty50_last_10_years.csv")),
    sp500: createMarketState("sp500", "S&P 500", publicAsset("sp500_last_10_years.csv")),
};

for (const market of Object.values(marketStates)) {
    market.trainSuccessGraph.render();
    market.validationSuccessGraph.render();
    market.errorGraph.render();
    market.interpretationGraph.render();
    if (market.haltButton) market.haltButton.disabled = true;
    setMarketStatus(market, `${market.label} selected. Click Train.`);
}

updateHiddenConfigVisibility();
cfgHiddenLayers?.addEventListener("input", () => updateHiddenConfigVisibility());
cfgBatchSize?.addEventListener("input", () => syncStepDefaultsFromBatchSize());
syncStepDefaultsFromBatchSize();

wireMarketActions(marketStates.nifty);
wireMarketActions(marketStates.sp500);

function createMarketState(market: MarketKey, label: string, csvPath: string): MarketState {
    const suffix = market === "nifty" ? "Nifty" : "Sp500";
    return {
        label,
        csvPath,
        debug: document.getElementById(`debug${suffix}`),
        crossDebug: document.getElementById(`crossDebug${suffix}`),
        trainButton: document.getElementById(`train${suffix}`) as HTMLInputElement | null,
        haltButton: document.getElementById(`halt${suffix}`) as HTMLInputElement | null,
        generateButton: document.getElementById(`generate${suffix}`) as HTMLInputElement | null,
        crossTestButton: document.getElementById(`crossTest${suffix}`) as HTMLInputElement | null,
        trainSuccessGraph: new SuccessGraph(
            document.getElementById(`successCanvas${suffix}`) as HTMLCanvasElement | null,
            { title: `${label} Train Accuracy`, xAxisLabel: "Epoch" },
        ),
        validationSuccessGraph: new SuccessGraph(
            document.getElementById(`validationCanvas${suffix}`) as HTMLCanvasElement | null,
            { title: `${label} Validation Accuracy`, xAxisLabel: "Epoch/Run" },
        ),
        errorGraph: new ErrorGraph(document.getElementById(`errorCanvas${suffix}`) as HTMLCanvasElement | null),
        interpretationGraph: new InterpretationGraph(
            document.getElementById(`interpretationCanvas${suffix}`) as HTMLCanvasElement | null,
        ),
        model: null,
        previewX: null,
        previewY: null,
        lastTrainingConfig: null,
        isTraining: false,
        haltRequested: false,
    };
}

function wireMarketActions(market: MarketState) {
    market.trainButton?.addEventListener('click', async () => {
        if (market.isTraining) return;
        market.isTraining = true;
        market.haltRequested = false;
        if (market.trainButton) market.trainButton.disabled = true;
        if (market.haltButton) market.haltButton.disabled = false;
        if (market.generateButton) market.generateButton.disabled = true;
        setMarketStatus(market, `Training ${market.label}...`);

        try {
            await trainMarket(market);
        } catch (error) {
            const haltedByUser = error instanceof Error && error.message === TRAIN_HALTED_ERROR;
            if (haltedByUser) setMarketStatus(market, `${market.label} training halted.`);
            else {
                console.error(`Failed to load/train from ${market.label} CSV:`, error);
                setMarketStatus(market, `${market.label} training failed (check console).`);
            }
        } finally {
            market.isTraining = false;
            market.haltRequested = false;
            if (market.trainButton) market.trainButton.disabled = false;
            if (market.haltButton) market.haltButton.disabled = true;
            if (market.generateButton) market.generateButton.disabled = false;
            if (market.model && market.previewX && market.previewY) {
                setMarketStatus(market, `${market.label} model ready.`);
            }
        }
    });

    market.haltButton?.addEventListener('click', () => {
        if (!market.isTraining) return;
        market.haltRequested = true;
        market.haltButton!.disabled = true;
        setMarketStatus(market, `Halting ${market.label} after current batch...`);
    });

    market.generateButton?.addEventListener('click', () => generateForMarket(market));
    market.crossTestButton?.addEventListener('click', async () => {
        if (market.isTraining) return;
        await crossTestMarket(market);
    });
}

async function trainMarket(market: MarketState) {
    logTfMemory(`Before ${market.label} training load`);
    const ui = readTrainingConfig();
    const dataset = await loadNiftyTrainingSet(
        market.csvPath,
        ui.batchSize,
        ui.validationRatio,
        ui.lookbackDays,
        ui.datasetMode,
        ui.smoothingWindow,
        ui.predictionHorizonDays,
        ui.flatThresholdPct,
    );
    let nextModel: TrainableModel | null = null;
    let trainingCompleted = false;

    try {
        const shuffledTrainBatches = shuffledBatches(dataset.trainBatches);
        const setup: PCNConfig = {
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
            stopRequested: () => market.haltRequested,
            trainEvalData: { x: dataset.trainX, y: dataset.trainY },
            validationEvalData: { x: dataset.validationX, y: dataset.validationY },
            data: shuffledTrainBatches,
        };

        const dims = [
            dataset.inputSize,
            ...buildHiddenSizes(ui.hiddenLayerCount, ui.hiddenSizes),
            dataset.outputSize,
        ];
        nextModel = ui.modelKind === "standard" ? new StandardClassifier(dims) : new PCN(dims);

        console.log(
            `Loaded ${dataset.sampleCount} samples from ${market.label} CSV ` +
            `(train=${dataset.trainCount}, validation=${dataset.validationCount}).`,
        );
        console.log(`Input context -> ${ui.lookbackDays} day(s), ${dataset.inputSize} features per sample.`);
        console.log(`Model type -> ${ui.modelKind === "standard" ? "standard tfjs mlp" : "pcn"}`);
        console.log(`Dataset mode -> ${ui.datasetMode === "sma" ? `SMA (${ui.smoothingWindow})` : "raw"}`);
        console.log(`Target -> ${ui.predictionHorizonDays}-day forward move, flat band +/-${ui.flatThresholdPct}%`);
        console.log(
            `Validation baselines -> majority=${(dataset.baselines.validationMajorityAccuracy * 100).toFixed(2)}%, ` +
            `persistence=${(dataset.baselines.validationPersistenceAccuracy * 100).toFixed(2)}%`,
        );
        console.log(`PCN architecture -> ${dims.join(" -> ")}`);

        const report = await nextModel.Train(setup);
        if (market.haltRequested) throw new Error(TRAIN_HALTED_ERROR);

        market.model?.dispose();
        market.model = nextModel;
        trainingCompleted = true;

        market.previewX?.dispose();
        market.previewY?.dispose();
        market.previewX = dataset.validationX.clone();
        market.previewY = dataset.validationY.clone();
        market.lastTrainingConfig = { ...ui, hiddenSizes: [...ui.hiddenSizes] };

        market.errorGraph.setHistory(report.epochMSE);
        market.trainSuccessGraph.setHistory(report.epochTrainAccuracy);
        market.validationSuccessGraph.setHistory(report.epochValidationAccuracy);
        market.validationSuccessGraph.setBaselines([
            { label: "Majority", value: dataset.baselines.validationMajorityAccuracy, color: "#f59e0b" },
            { label: "Persistence", value: dataset.baselines.validationPersistenceAccuracy, color: "#6366f1" },
        ]);

        const postTrainSuccess = evaluateAndRenderMarket(market, market.previewX, market.previewY, ui);
        market.validationSuccessGraph.record(postTrainSuccess);
        setMarketStatus(
            market,
            `${market.label} post-train validation: ${(postTrainSuccess * 100).toFixed(2)}% ` +
            `(majority ${(dataset.baselines.validationMajorityAccuracy * 100).toFixed(2)}%)`,
        );
    } finally {
        disposeDataset(dataset);
        if (!trainingCompleted) nextModel?.dispose();
        logTfMemory(`After ${market.label} training cleanup`);
    }
}

function generateForMarket(market: MarketState) {
    if (!market.model || !market.previewX || !market.previewY) {
        console.warn(`${market.label} model is not trained yet. Click Train first.`);
        setMarketStatus(market, `Train the ${market.label} model first.`);
        return;
    }

    const ui = market.lastTrainingConfig ?? readTrainingConfig();
    const success = evaluateAndRenderMarket(market, market.previewX, market.previewY, ui);
    market.validationSuccessGraph.record(success);
    logTfMemory(`After ${market.label} generate`);
    setMarketStatus(market, `${market.label} generate success: ${(success * 100).toFixed(2)}%`);
}

async function crossTestMarket(sourceMarket: MarketState) {
    const targetMarket = getOtherMarket(sourceMarket);
    if (!sourceMarket.model) {
        setCrossStatus(sourceMarket, `Train ${sourceMarket.label} before running a cross-test.`);
        return;
    }
    if (!sourceMarket.lastTrainingConfig) {
        setCrossStatus(sourceMarket, `Missing saved training settings for ${sourceMarket.label}. Train again first.`);
        return;
    }

    const ui = sourceMarket.lastTrainingConfig;
    setCrossStatus(sourceMarket, `Testing ${sourceMarket.label} model on ${targetMarket.label} data...`);
    if (sourceMarket.crossTestButton) sourceMarket.crossTestButton.disabled = true;
    let dataset: NiftyTrainingSet | null = null;
    try {
        dataset = await loadNiftyTrainingSet(
            targetMarket.csvPath,
            ui.batchSize,
            ui.validationRatio,
            ui.lookbackDays,
            ui.datasetMode,
            ui.smoothingWindow,
            ui.predictionHorizonDays,
            ui.flatThresholdPct,
        );
        const success = evaluateAndRenderMarket(sourceMarket, dataset.validationX, dataset.validationY, ui);
        setCrossStatus(
            sourceMarket,
            `${sourceMarket.label} on ${targetMarket.label}: ${(success * 100).toFixed(2)}% ` +
            `(majority ${(dataset.baselines.validationMajorityAccuracy * 100).toFixed(2)}%, ` +
            `persistence ${(dataset.baselines.validationPersistenceAccuracy * 100).toFixed(2)}%)`,
        );
    } catch (error) {
        console.error(`Cross-test failed for ${sourceMarket.label} on ${targetMarket.label}:`, error);
        setCrossStatus(sourceMarket, `Cross-test failed for ${sourceMarket.label} on ${targetMarket.label}.`);
    } finally {
        if (dataset) disposeDataset(dataset);
        if (sourceMarket.crossTestButton) sourceMarket.crossTestButton.disabled = false;
    }
}

function evaluateAndRenderMarket(
    market: MarketState,
    x: tf.Tensor2D,
    y: tf.Tensor2D,
    ui: Pick<TrainingUiConfig, 'T_infer' | 'eta_infer'>,
): number {
    if (!market.model) return 0;
    const pred = market.model.GenerateMapped(x, ui.T_infer, ui.eta_infer);
    const predClass = classIds(pred);
    const targetClass = classIds(y);
    market.interpretationGraph.setFromClasses(predClass, targetClass, [...MARKET_CLASS_LABELS]);
    const success = classAccuracy(predClass, targetClass);
    pred.dispose();
    return success;
}

function disposeDataset(dataset: NiftyTrainingSet) {
    for (const batch of dataset.trainBatches) {
        batch.x.dispose();
        batch.y.dispose();
    }
    dataset.trainX.dispose();
    dataset.trainY.dispose();
    dataset.validationX.dispose();
    dataset.validationY.dispose();
}

function setMarketStatus(market: MarketState, text: string) {
    if (market.debug) market.debug.innerText = text;
}

function setCrossStatus(market: MarketState, text: string) {
    if (market.crossDebug) market.crossDebug.innerText = text;
}

function getOtherMarket(market: MarketState): MarketState {
    return market === marketStates.nifty ? marketStates.sp500 : marketStates.nifty;
}

function classIds(mat: tf.Tensor2D): Int32Array {
    const classes = tf.argMax(mat, 1);
    const ids = Int32Array.from(classes.dataSync());
    classes.dispose();
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

function readTrainingConfig(): TrainingUiConfig {
    const batchSize = intOrDefault(cfgBatchSize, 128, 1);
    return {
        modelKind: cfgModelKind?.value === "standard" ? "standard" : "pcn",
        datasetMode: cfgDatasetMode?.value === "sma" ? "sma" : "raw",
        smoothingWindow: intOrDefault(cfgSmoothingWindow, 5, 2, 30),
        predictionHorizonDays: intOrDefault(cfgPredictionHorizonDays, 3, 1, 30),
        flatThresholdPct: numberOrDefault(cfgFlatThresholdPct, 0.5, 0, 10),
        useSignedOutputMapping: checkboxOrDefault(cfgSignedOutput, false),
        haltOnNonFinite: checkboxOrDefault(cfgHaltNonFinite, true),
        pcnDebug: checkboxOrDefault(cfgDebug, false),
        batchSize,
        lookbackDays: intOrDefault(cfgLookbackDays, 5, 1, 60),
        validationRatio: numberOrDefault(cfgValidationRatio, 0.2, 0.05, 0.45),
        epochs: intOrDefault(cfgEpochs, 40, 1),
        T_infer: intOrDefault(cfgInferSteps, defaultInferSteps(batchSize), 1),
        T_learn: intOrDefault(cfgLearnSteps, defaultLearnSteps(batchSize), 1),
        eta_infer: numberOrDefault(cfgInferLr, 0.05, 1e-6),
        eta_learn: numberOrDefault(cfgLearnLr, 0.005, 1e-7),
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

function defaultLearnSteps(batchSize: number): number {
    return Math.max(1, Math.floor(batchSize));
}

function defaultInferSteps(batchSize: number): number {
    return Math.max(1, Math.round(batchSize / 10));
}

function syncStepDefaultsFromBatchSize() {
    const batchSize = intOrDefault(cfgBatchSize, 128, 1);
    if (cfgLearnSteps) cfgLearnSteps.value = String(defaultLearnSteps(batchSize));
    if (cfgInferSteps) cfgInferSteps.value = String(defaultInferSteps(batchSize));
}

function updateHiddenConfigVisibility() {
    const count = intOrDefault(cfgHiddenLayers, 2, 1, 4);
    const rows = document.querySelectorAll<HTMLElement>("[data-hidden-index]");
    for (const row of rows) {
        const idx = Number(row.dataset.hiddenIndex ?? "0");
        row.style.display = idx >= 1 && idx <= count ? "" : "none";
    }
}

function shuffledBatches<T>(items: T[]): T[] {
    const out = items.slice();
    for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [out[i], out[j]] = [out[j]!, out[i]!];
    }
    return out;
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
