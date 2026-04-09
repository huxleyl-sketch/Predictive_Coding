import * as tf from '@tensorflow/tfjs';

export type PCNConfig = {
    epochs: number;
    T_infer: number;
    eta_infer: number;
    T_learn: number;
    eta_learn: number;
    /** Toggle signed output mapping: tanh(output) and [-1,1] mapped target error */
    useSignedOutputMapping?: boolean;
    /** Clip gradient values to [-gradientClipValue, gradientClipValue] to reduce exploding updates */
    gradientClipValue?: number;
    /** Stop training early when a non-finite batch/epoch error is detected */
    haltOnNonFinite?: boolean;
    /** Track train/validation accuracy history after each epoch */
    collectAccuracyHistory?: boolean;
    /** Optional train set used for epoch-end accuracy tracking */
    trainEvalData?: {
        x: tf.Tensor2D;
        y: tf.Tensor2D;
    };
    /** Optional validation set used for epoch-end accuracy tracking */
    validationEvalData?: {
        x: tf.Tensor2D;
        y: tf.Tensor2D;
    };
    /** Inference steps used during epoch-end accuracy evaluation */
    evalInferSteps?: number;
    /** Enable verbose timing/logging for training diagnostics */
    debug?: boolean;
    data: {
        x: tf.Tensor2D;
        y: tf.Tensor2D;
    }[];
};

export type PCNTrainReport = {
    epochMSE: number[];
    epochTrainAccuracy: number[];
    epochValidationAccuracy: number[];
};

export class PCN {
    /** Latents */
    X: tf.Tensor2D[];
    /** Weights */
    W: tf.Variable[];
    /** Activation Function */
    f: (m: tf.Tensor2D) => tf.Tensor2D;
    /** Derivative of Activation Function */
    df: (m: tf.Tensor2D) => tf.Tensor2D;
    /** layers */
    L: number;
    /** Dimensions */
    D: number[];
    /** Error */
    E: tf.Tensor2D[];
    /** Gain Modulated Error */
    H: tf.Tensor2D[];
    /** Signed output mapping feature flag */
    useSignedOutputMapping: boolean;
    /** Debug logging/timing flag */
    debug: boolean;

    /**
     * @param dims Dimensions
     */
    constructor(dims: number[]) {
        this.f = (m: tf.Tensor2D) => m.relu();
        this.df = (m: tf.Tensor2D) => m.greater(0).toFloat() as tf.Tensor2D;

        this.H = [];
        this.E = [];
        this.X = [];
        this.W = [];
        this.useSignedOutputMapping = true;
        this.debug = false;

        this.D = dims;
        this.L = dims.length - 2;
    }

    private keepAssign(arr: tf.Tensor2D[], index: number, value: tf.Tensor2D) {
        const prev = arr[index];
        arr[index] = tf.keep(value) as tf.Tensor2D;
        prev?.dispose();
    }

    private log(message: string) {
        if (this.debug) console.log(message);
    }

    private warn(message: string) {
        if (this.debug) console.warn(message);
    }

    private time(label: string) {
        if (this.debug) console.time(label);
    }

    private timeEnd(label: string) {
        if (this.debug) console.timeEnd(label);
    }

    Train(input: PCNConfig = {
        epochs: 2000,
        T_infer: 50,
        eta_infer: 0.05,
        T_learn: 500,
        eta_learn: 0.005,
        data: [{
            x: tf.tensor2d([]),
            y: tf.tensor2d([]),
        }],
    }): PCNTrainReport {
        this.time("Training");
        const epochMSE: number[] = [];
        const epochTrainAccuracy: number[] = [];
        const epochValidationAccuracy: number[] = [];
        const T_infer = input.T_infer;
        const eta_infer = input.eta_infer;
        const T_learn = input.T_learn;
        const eta_learn = input.eta_learn;
        this.useSignedOutputMapping = input.useSignedOutputMapping ?? true;
        const gradientClipValue = input.gradientClipValue ?? 5;
        const haltOnNonFinite = input.haltOnNonFinite ?? true;
        const collectAccuracyHistory = input.collectAccuracyHistory ?? false;
        const evalInferSteps = input.evalInferSteps ?? T_infer;
        this.debug = input.debug ?? false;
        for (let l = 0; l < this.L; l++) {
            this.W[l] = tf.variable(xavierUniform(this.D[l]!, this.D[l + 1]!));
        }
        this.W[this.L + 1] = tf.variable(xavierUniform(this.D[this.L + 1]!, this.D[this.L]!));

        let shouldStop = false;
        for (let epoch = 0; epoch < input.epochs; epoch++) {
            let epochWeightedMSE = 0;
            let epochSamples = 0;
            for (const batch of input.data) {
                /** Batch Size */
                const B = batch.x.shape[0]!;
                const negInvB = tf.scalar(-1 / B);
                const invB = tf.scalar(1 / B);
                this.H[0] = zeros(B, this.D[0]!);
                this.H[this.L + 1] = zeros(B, this.D[this.L + 1]!);

                /** Initialise Values */
                for (let l = 1; l <= this.L; l++) {
                    /** Small Random Values */
                    this.X[l] = xavierUniform(B, this.D[l]!);
                }

                /** Input Batch Fixing */
                this.X[0] = batch.x;
                const Y = batch.y;
                this.time("Inference");
                /** Inference Update Loop */
                for (let t = 1; t <= T_infer; t++) {
                    tf.tidy(() => {
                        for (let l = 0; l < this.L; l++) {
                            /** Pre Activation Predicted Latent Values */
                            const A_l = tf.matMul(this.X[l + 1], this.W[l], false, true) satisfies tf.Tensor2D;
                            /** Predicted Latent Values */
                            const Xhat_l = this.f(A_l);
                            this.keepAssign(this.E, l, tf.sub(this.X[l], Xhat_l));
                            this.keepAssign(this.H, l, tf.mul(this.E[l], this.df(A_l)));
                        }

                        const Yhat = tf.matMul(this.X[this.L], this.W[this.L + 1], false, true);

                        const E_sup = tf.sub(Yhat, Y);
                        this.keepAssign(this.E, this.L, tf.matMul(E_sup, this.W[this.L + 1]));

                        for (let l = 1; l <= this.L; l++) {
                            /** Gradients of Latents */
                            const G_xlRaw = tf.sub(this.E[l], tf.matMul(this.H[l - 1], this.W[l - 1]));
                            const G_xl = tf.clipByValue(G_xlRaw, -gradientClipValue, gradientClipValue);
                            this.keepAssign(this.X, l, tf.sub(this.X[l], tf.mul(eta_infer, G_xl)));
                        }
                    });
                }
                this.timeEnd("Inference");
                this.time("Weights");
                /** Weight Update Loop */
                for (let t = 1; t <= T_learn; t++) {
                    tf.tidy(() => {
                        for (let l = 0; l < this.L; l++) {
                            const A_l = tf.matMul(this.X[l + 1], this.W[l], false, true) satisfies tf.Tensor2D;
                            const Xhat_l = this.f(A_l);
                            this.keepAssign(this.E, l, tf.sub(this.X[l], Xhat_l));
                            this.keepAssign(this.H, l, tf.mul(this.E[l], this.df(A_l)));
                            /** Average Gradients of Weights */
                            const G_wlRaw = tf.mul(negInvB, tf.matMul(this.H[l].transpose(), this.X[l + 1]));
                            const G_wl = tf.clipByValue(G_wlRaw, -gradientClipValue, gradientClipValue);
                            this.W[l].assign(tf.sub(this.W[l], tf.mul(eta_learn, G_wl)));
                        }
                        const Yhat = tf.matMul(this.X[this.L], this.W[this.L + 1], false, true);
                        const E_sup = tf.sub(Yhat, Y);
                        /** Average Gradients of output weight */
                        const G_w_outRaw = tf.mul(invB, tf.matMul(E_sup.transpose(), this.X[this.L]));
                        const G_w_out = tf.clipByValue(G_w_outRaw, -gradientClipValue, gradientClipValue);
                        this.W[this.L + 1].assign(tf.sub(this.W[this.L + 1], tf.mul(eta_learn, G_w_out)));
                    });
                }
                this.timeEnd("Weights");

                const batchMSE = tf.tidy(() => {
                    const Yhat = tf.matMul(this.X[this.L], this.W[this.L + 1], false, true);
                    const mse = this.useSignedOutputMapping
                        ? tf.mean(tf.square(tf.sub(tf.tanh(Yhat), mapTargetToSigned(Y))))
                        : tf.mean(tf.square(tf.sub(Yhat, Y)));
                    return mse.dataSync()[0] ?? 0;
                });
                if (!Number.isFinite(batchMSE)) {
                    this.warn(`Non-finite batch MSE detected at epoch ${epoch + 1}.`);
                    shouldStop = true;
                }
                epochWeightedMSE += batchMSE * B;
                epochSamples += B;

                negInvB.dispose();
                invB.dispose();
                if (shouldStop) break;
            }
            const meanEpochMSE = epochSamples > 0 ? epochWeightedMSE / epochSamples : 0;
            if (!Number.isFinite(meanEpochMSE)) {
                this.warn(`Non-finite epoch MSE at epoch ${epoch + 1}.`);
                shouldStop = true;
            }
            epochMSE.push(meanEpochMSE);
            this.log(`${epoch + 1} / ${input.epochs} | mse=${meanEpochMSE.toFixed(6)}`);
            if (collectAccuracyHistory) {
                if (input.trainEvalData) {
                    epochTrainAccuracy.push(
                        this.evaluateAccuracy(input.trainEvalData.x, input.trainEvalData.y, evalInferSteps, eta_infer),
                    );
                }
                if (input.validationEvalData) {
                    epochValidationAccuracy.push(
                        this.evaluateAccuracy(input.validationEvalData.x, input.validationEvalData.y, evalInferSteps, eta_infer),
                    );
                }
            }
            if (shouldStop && haltOnNonFinite) {
                this.warn("Stopping early due to non-finite training values.");
                break;
            }
        }
        this.timeEnd("Training");
        return { epochMSE, epochTrainAccuracy, epochValidationAccuracy };
    }

    Generate(xBatch: tf.Tensor2D, T_infer: number, eta_infer: number) {
        this.X[0] = xBatch;
        for (let l = 1; l <= this.L; l++) {
            /** Small Random Values */
            this.X[l] = xavierUniform(xBatch.shape[0], this.D[l]!);
        }
        /** Inference Update Loop */
        for (let t = 1; t <= T_infer; t++) {
            tf.tidy(() => {
                for (let l = 0; l < this.L; l++) {
                    /** Pre Activation Predicted Latent Values */
                    const A_l = tf.matMul(this.X[l + 1], this.W[l], false, true) satisfies tf.Tensor2D;
                    /** Predicted Latent Values */
                    const Xhat_l = this.f(A_l);
                    this.keepAssign(this.E, l, tf.sub(this.X[l], Xhat_l));
                    this.keepAssign(this.H, l, tf.mul(this.E[l], this.df(A_l)));
                }
                this.keepAssign(this.E, this.L, zeros(xBatch.shape[0]!, this.D[this.L]!));

                for (let l = 1; l <= this.L; l++) {
                    /** Gradients of Latents */
                    const G_xl = tf.sub(this.E[l], tf.matMul(this.H[l - 1], this.W[l - 1]));
                    this.keepAssign(this.X, l, tf.sub(this.X[l], tf.mul(eta_infer, G_xl)));
                }
            });
        }
        return tf.matMul(this.X[this.L], this.W[this.L + 1], false, true) satisfies tf.Tensor2D;
    }

    GenerateMapped(
        xBatch: tf.Tensor2D,
        T_infer: number,
        eta_infer: number,
        useSignedOutputMapping: boolean = this.useSignedOutputMapping,
    ) {
        const raw = this.Generate(xBatch, T_infer, eta_infer);
        if (!useSignedOutputMapping) return raw;
        return tf.tanh(raw) satisfies tf.Tensor2D;
    }

    private evaluateAccuracy(
        xBatch: tf.Tensor2D,
        target: tf.Tensor2D,
        inferSteps: number,
        etaInfer: number,
    ): number {
        const prediction = this.GenerateMapped(xBatch, inferSteps, etaInfer);
        const score = tensorAccuracy(prediction, target);
        prediction.dispose();
        return score;
    }
}

function zeros(rows: number, cols: number): tf.Tensor2D {
    return tf.zeros([rows, cols]);
}

function xavierUniform(rows: number, cols: number): tf.Tensor2D {
    const limit = Math.sqrt(6 / (rows + cols));
    return tf.randomUniform([rows, cols], -limit, limit);
}

function mapTargetToSigned(target: tf.Tensor2D): tf.Tensor2D {
    // Maps [0,1] labels into [-1,1] so error can be measured on the same range.
    return tf.sub(tf.mul(target, 2), 1) as tf.Tensor2D;
}

function tensorAccuracy(predictions: tf.Tensor2D, target: tf.Tensor2D): number {
    const predictedClass = tf.argMax(predictions, 1);
    const targetClass = tf.argMax(target, 1);
    const accuracy = tf.mean(tf.cast(tf.equal(predictedClass, targetClass), "float32"));
    const score = accuracy.dataSync()[0] ?? 0;
    predictedClass.dispose();
    targetClass.dispose();
    accuracy.dispose();
    return score;
}
