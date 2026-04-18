import * as tf from '@tensorflow/tfjs';
import type { PCNConfig, PCNTrainReport } from './pcn';

type TrainableModel = tf.LayersModel;

export class StandardClassifier {
    private model: TrainableModel | null = null;
    private dims: number[];
    private debug = false;

    constructor(dims: number[]) {
        this.dims = dims;
    }

    async Train(input: PCNConfig): Promise<PCNTrainReport> {
        this.debug = input.debug ?? false;
        this.dispose();
        this.model = this.buildModel(input.eta_learn);

        const epochMSE: number[] = [];
        const epochTrainAccuracy: number[] = [];
        const epochValidationAccuracy: number[] = [];
        const stopRequested = input.stopRequested ?? (() => false);

        for (let epoch = 0; epoch < input.epochs; epoch++) {
            if (stopRequested()) break;

            let weightedLoss = 0;
            let sampleCount = 0;

            for (let batchIdx = 0; batchIdx < input.data.length; batchIdx++) {
                if (stopRequested()) break;
                const batch = input.data[batchIdx]!;
                const batchSize = batch.x.shape[0] ?? 0;
                const history = await this.model.trainOnBatch(batch.x, batch.y);
                const batchLoss = Array.isArray(history) ? Number(history[0] ?? 0) : Number(history ?? 0);
                weightedLoss += batchLoss * batchSize;
                sampleCount += batchSize;
                await tf.nextFrame();
            }

            const meanLoss = sampleCount > 0 ? weightedLoss / sampleCount : 0;
            epochMSE.push(meanLoss);
            if (this.debug) console.log(`${epoch + 1} / ${input.epochs} | loss=${meanLoss.toFixed(6)}`);

            if (input.collectAccuracyHistory) {
                if (input.trainEvalData) {
                    epochTrainAccuracy.push(this.evaluateAccuracy(input.trainEvalData.x, input.trainEvalData.y));
                }
                if (input.validationEvalData) {
                    epochValidationAccuracy.push(
                        this.evaluateAccuracy(input.validationEvalData.x, input.validationEvalData.y),
                    );
                }
            }
        }

        return { epochMSE, epochTrainAccuracy, epochValidationAccuracy };
    }

    GenerateMapped(xBatch: tf.Tensor2D, _T_infer: number, _eta_infer: number): tf.Tensor2D {
        if (!this.model) throw new Error("Standard model is not initialized.");
        return this.model.predict(xBatch) as tf.Tensor2D;
    }

    dispose() {
        this.model?.dispose();
        this.model = null;
    }

    private buildModel(learningRate: number): TrainableModel {
        const model = tf.sequential();
        const hiddenDims = this.dims.slice(1, -1);

        hiddenDims.forEach((units, idx) => {
            model.add(tf.layers.dense({
                units,
                activation: 'relu',
                inputShape: idx === 0 ? [this.dims[0]!] : undefined,
                kernelInitializer: 'glorotUniform',
            }));
        });

        model.add(tf.layers.dense({
            units: this.dims[this.dims.length - 1]!,
            activation: 'softmax',
            kernelInitializer: 'glorotUniform',
        }));

        model.compile({
            optimizer: tf.train.adam(Math.max(1e-6, learningRate)),
            loss: 'categoricalCrossentropy',
        });

        return model;
    }

    private evaluateAccuracy(xBatch: tf.Tensor2D, target: tf.Tensor2D): number {
        const prediction = this.GenerateMapped(xBatch, 0, 0);
        const predictedClass = tf.argMax(prediction, 1);
        const targetClass = tf.argMax(target, 1);
        const accuracy = tf.mean(tf.cast(tf.equal(predictedClass, targetClass), "float32"));
        const score = accuracy.dataSync()[0] ?? 0;
        prediction.dispose();
        predictedClass.dispose();
        targetClass.dispose();
        accuracy.dispose();
        return score;
    }
}
