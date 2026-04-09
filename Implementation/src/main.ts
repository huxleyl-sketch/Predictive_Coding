import * as tf from '@tensorflow/tfjs';
import { loadNiftyTrainingSet } from './niftyTrainingData';
import { PCN, type PCNConfig } from './pcn';
import { ErrorGraph, InterpretationGraph, SuccessGraph, computeSuccess } from './successGraph';
import './style.css';

// Ensure a backend is registered before any eager ops run.
import '@tensorflow/tfjs-backend-webgl';
await tf.setBackend('webgl');
await tf.ready();

const debug = document.getElementById("debug");
const trainButton = document.getElementById("train") as HTMLInputElement | null;
const generateButton = document.getElementById("retry") as HTMLInputElement | null;
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

let pcn: PCN | null = null;
let previewX: tf.Tensor2D | null = null;
let previewY: tf.Tensor2D | null = null;
let isTraining = false;

if (debug) debug.innerText = "Ready. Click Train.";

async function trainFromNiftyCsv() {
    const dataset = await loadNiftyTrainingSet('/nifty50_last_10_years.csv', 128, 0.2);
    // FEATURE_TOGGLE_SIGNED_OUTPUT_MAPPING: true => use tanh output in [-1,1], false => raw output space.
    const useSignedOutputMapping = true;
    // FEATURE_TOGGLE_GRADIENT_CLIP_VALUE: lower this (e.g. 1-3) if you still see unstable/NaN training.
    const gradientClipValue = 3;
    // FEATURE_TOGGLE_HALT_ON_NON_FINITE: stop early if NaN/Inf is detected.
    const haltOnNonFinite = true;
    // FEATURE_TOGGLE_PCN_DEBUG: true enables PCN console.log/time instrumentation.
    const pcnDebug = false;
    // PERFORMANCE_PRESET: stronger defaults than the initial quick smoke-test setup.
    const epochs = 40;
    const T_infer = 25;
    const T_learn = 25;
    const eta_infer = 0.01;
    const eta_learn = 0.0001;

    const pcnSetup: PCNConfig = {
        epochs,
        T_infer,
        eta_infer,
        T_learn,
        eta_learn,
        useSignedOutputMapping,
        gradientClipValue,
        haltOnNonFinite,
        debug: pcnDebug,
        collectAccuracyHistory: true,
        trainEvalData: { x: dataset.trainX, y: dataset.trainY },
        validationEvalData: { x: dataset.validationX, y: dataset.validationY },
        evalInferSteps: 40,
        data: dataset.trainBatches,
    };

    previewX = dataset.previewX;
    previewY = dataset.previewY;

    pcn = new PCN([dataset.inputSize, 16, 8, dataset.outputSize]);
    console.log(
        `Loaded ${dataset.sampleCount} samples from NIFTY50 CSV ` +
        `(train=${dataset.trainCount}, validation=${dataset.validationCount}).`,
    );
    console.log(
        `Validation baselines -> majority=${(dataset.baselines.validationMajorityAccuracy * 100).toFixed(2)}%, ` +
        `persistence=${(dataset.baselines.validationPersistenceAccuracy * 100).toFixed(2)}%`,
    );

    const trainReport = pcn.Train(pcnSetup);
    errorGraph.setHistory(trainReport.epochMSE);
    trainSuccessGraph.setHistory(trainReport.epochTrainAccuracy);
    validationSuccessGraph.setHistory(trainReport.epochValidationAccuracy);
    validationSuccessGraph.setBaselines([
        { label: "Majority", value: dataset.baselines.validationMajorityAccuracy, color: "#f59e0b" },
        { label: "Persistence", value: dataset.baselines.validationPersistenceAccuracy, color: "#6366f1" },
    ]);

    const postTrain = pcn.GenerateMapped(previewX, 50, 0.05);
    const postTrainClass = classIds(postTrain);
    const postTrainTargetClass = classIds(previewY);
    interpretationGraph.setFromClasses(postTrainClass, postTrainTargetClass);
    const postTrainSuccess = computeSuccess(postTrain, previewY);
    validationSuccessGraph.record(postTrainSuccess);
    postTrain.dispose();

    if (debug) {
        debug.innerText =
            `Post-train validation: ${(postTrainSuccess * 100).toFixed(2)}% ` +
            `(majority ${(dataset.baselines.validationMajorityAccuracy * 100).toFixed(2)}%)`;
    }
}

try {
    trainButton?.addEventListener('click', async () => {
        if (isTraining) return;
        isTraining = true;
        if (trainButton) trainButton.disabled = true;
        if (debug) debug.innerText = "Training...";

        try {
            await trainFromNiftyCsv();
        } catch (error) {
            console.error('Failed to load/train from NIFTY50 CSV:', error);
            if (debug) debug.innerText = "Training failed (check console)";
        } finally {
            isTraining = false;
            if (trainButton) trainButton.disabled = false;
        }
    });
} catch (error) {
    console.error('Failed to initialize controls:', error);
    if (debug) debug.innerText = "UI init failed (check console)";
}

generateButton?.addEventListener('click', () => {
    if (!pcn || !previewX || !previewY) {
        console.warn("Model is not trained yet. Click Train first.");
        if (debug) debug.innerText = "Train the model first.";
        return;
    }

    const result = pcn.GenerateMapped(previewX, 50, 0.05);
    const resultClass = classIds(result);
    const targetClass = classIds(previewY);
    interpretationGraph.setFromClasses(resultClass, targetClass);
    const success = validationSuccessGraph.record(computeSuccess(result, previewY));

    //console.log("Result:", Array.from(resultClass));
    //console.log("Target:", Array.from(targetClass));
    result.dispose();

    if (debug) debug.innerText = `Generate success: ${(success * 100).toFixed(2)}%`;
});

function classIds(mat: tf.Tensor2D): Int32Array {
    const classes = tf.argMax(mat, 1);
    const ids = Int32Array.from(classes.dataSync());
    classes.dispose();
    return ids;
}
