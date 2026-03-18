export {}

/**
 * Supervised Predictive Coding Network (PCN) in TypeScript
 *
 * This is a small, annotated implementation of the supervised learning algorithm
 * described in the uploaded paper. It mirrors the vectorized row-batch form:
 *
 *   A^(l)   = X^(l+1) W^(l)^T
 *   Xhat^(l)= f(A^(l))
 *   E^(l)   = X^(l) - Xhat^(l)
 *   H^(l)   = E^(l) ⊙ f'(A^(l))
 *   Yhat    = X^(L) Wout^T
 *   Esup    = Yhat - Y
 *   E^(L)   = Esup Wout
 *
 * Inference:
 *   Gx^(l)  = E^(l) - H^(l-1) W^(l-1)
 *   X^(l)   = X^(l) - etaInfer * Gx^(l)
 *
 * Learning:
 *   Gw^(l)  = -(1/B) H^(l)^T X^(l+1)
 *   W^(l)   = W^(l) - etaLearn * Gw^(l)
 *
 *   Gwout   = (1/B) Esup^T X^(L)
 *   Wout    = Wout - etaLearn * Gwout
 *
 * Shapes:
 *   X^(l):   [B, d_l]
 *   W^(l):   [d_l, d_(l+1)]
 *   Wout:    [d_out, d_L]
 *
 * This implementation uses plain arrays for clarity.
 */

type Matrix = number[][];
type Vector = number[];

interface Activation {
  fn: (x: number) => number;
  deriv: (x: number) => number;
}

interface PCNConfig {
  dims: number[];          // [d0, d1, ..., dL]
  outputDim: number;       // d_out
  etaInfer: number;
  etaLearn: number;
  tInfer: number;
  tLearn: number;
  activation?: Activation; // defaults to ReLU
  latentInitStd?: number;  // defaults to 0.01
}

interface ForwardCache {
  A: Matrix[];      // preactivations for layers 0..L-1
  XHat: Matrix[];   // predictions for layers 0..L-1
  E: Matrix[];      // prediction errors for layers 0..L-1
  H: Matrix[];      // gain-modulated errors for layers 0..L-1
  YHat: Matrix;     // [B, outputDim]
  ESup: Matrix;     // [B, outputDim]
  ETop: Matrix;     // E^(L) = ESup * Wout, shape [B, dL]
}

interface TrainStepResult {
  logits: Matrix;
  loss: number;
  latents: Matrix[];   // X^(1)..X^(L)
}

/* -------------------------------------------------------------------------- */
/*                                Math helpers                                */
/* -------------------------------------------------------------------------- */

function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function cloneMatrix(m: Matrix): Matrix {
  return m.map(row => [...row]);
}

function transpose(m: Matrix): Matrix {
  const rows = m.length;
  const cols = m[0].length;
  const out = zeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j][i] = m[i][j];
    }
  }
  return out;
}

function matMul(a: Matrix, b: Matrix): Matrix {
  const aRows = a.length;
  const aCols = a[0].length;
  const bRows = b.length;
  const bCols = b[0].length;

  if (aCols !== bRows) {
    throw new Error(`matMul shape mismatch: [${aRows},${aCols}] x [${bRows},${bCols}]`);
  }

  const out = zeros(aRows, bCols);
  for (let i = 0; i < aRows; i++) {
    for (let k = 0; k < aCols; k++) {
      const aik = a[i][k];
      for (let j = 0; j < bCols; j++) {
        out[i][j] += aik * b[k][j];
      }
    }
  }
  return out;
}

function add(a: Matrix, b: Matrix): Matrix {
  const out = zeros(a.length, a[0].length);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      out[i][j] = a[i][j] + b[i][j];
    }
  }
  return out;
}

function sub(a: Matrix, b: Matrix): Matrix {
  const out = zeros(a.length, a[0].length);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      out[i][j] = a[i][j] - b[i][j];
    }
  }
  return out;
}

function scale(a: Matrix, s: number): Matrix {
  const out = zeros(a.length, a[0].length);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      out[i][j] = a[i][j] * s;
    }
  }
  return out;
}

function hadamard(a: Matrix, b: Matrix): Matrix {
  const out = zeros(a.length, a[0].length);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      out[i][j] = a[i][j] * b[i][j];
    }
  }
  return out;
}

function mapMatrix(a: Matrix, fn: (x: number) => number): Matrix {
  const out = zeros(a.length, a[0].length);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      out[i][j] = fn(a[i][j]);
    }
  }
  return out;
}

function meanSquaredHalf(a: Matrix): number {
  let sum = 0;
  let count = 0;
  for (const row of a) {
    for (const x of row) {
      sum += x * x;
      count++;
    }
  }
  return 0.5 * sum / Math.max(count, 1);
}

/* -------------------------------------------------------------------------- */
/*                         Random init / utility helpers                      */
/* -------------------------------------------------------------------------- */

function randn(): number {
  // Box-Muller transform
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randomMatrix(rows: number, cols: number, std = 1): Matrix {
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[i][j] = randn() * std;
    }
  }
  return out;
}

function xavierUniform(rows: number, cols: number): Matrix {
  // For W shape [fanOut, fanIn]
  const limit = Math.sqrt(6 / (rows + cols));
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[i][j] = (Math.random() * 2 - 1) * limit;
    }
  }
  return out;
}

function oneHot(labels: number[], numClasses: number): Matrix {
  const out = zeros(labels.length, numClasses);
  for (let i = 0; i < labels.length; i++) {
    out[i][labels[i]] = 1;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/*                           Predictive Coding Network                        */
/* -------------------------------------------------------------------------- */

class PredictiveCodingNetwork {
  dims: number[];
  outputDim: number;
  L: number;
  etaInfer: number;
  etaLearn: number;
  tInfer: number;
  tLearn: number;
  activation: Activation;
  latentInitStd: number;

  // Generative weights W^(l): shape [d_l, d_(l+1)]
  weights: Matrix[];

  // Readout weights Wout: shape [d_out, d_L]
  wOut: Matrix;

  constructor(config: PCNConfig) {
    if (config.dims.length < 2) {
      throw new Error("dims must include at least input and one latent layer");
    }

    this.dims = config.dims;
    this.outputDim = config.outputDim;
    this.L = config.dims.length - 1;
    this.etaInfer = config.etaInfer;
    this.etaLearn = config.etaLearn;
    this.tInfer = config.tInfer;
    this.tLearn = config.tLearn;
    this.latentInitStd = config.latentInitStd ?? 0.01;

    this.activation = config.activation ?? {
      fn: (x: number) => Math.max(0, x),          // ReLU
      deriv: (x: number) => (x > 0 ? 1 : 0),      // d/dx ReLU
    };

    // Initialize generative weights W^(0)...W^(L-1)
    this.weights = [];
    for (let l = 0; l < this.L; l++) {
      const dLower = this.dims[l];
      const dUpper = this.dims[l + 1];
      this.weights.push(xavierUniform(dLower, dUpper));
    }

    // Initialize readout Wout
    this.wOut = xavierUniform(this.outputDim, this.dims[this.L]);
  }

  /**
   * Initialize latent states X^(1)...X^(L) with small Gaussian noise.
   * This follows the paper's setup: latents are freshly initialized per batch/sample.
   */
  initLatents(batchSize: number): Matrix[] {
    const latents: Matrix[] = [];
    for (let l = 1; l <= this.L; l++) {
      latents.push(randomMatrix(batchSize, this.dims[l], this.latentInitStd));
    }
    return latents;
  }

  /**
   * Compute the per-layer quantities:
   *   A^(l), Xhat^(l), E^(l), H^(l)
   * plus:
   *   Yhat, ESup, ETop
   *
   * inputsLatents is [X^(0), X^(1), ..., X^(L)]
   */
  computeForwardQuantities(inputsLatents: Matrix[], targets: Matrix): ForwardCache {
    const A: Matrix[] = [];
    const XHat: Matrix[] = [];
    const E: Matrix[] = [];
    const H: Matrix[] = [];

    // For each generative layer l = 0..L-1:
    // A^(l) = X^(l+1) W^(l)^T
    for (let l = 0; l < this.L; l++) {
      const XAbove = inputsLatents[l + 1];               // X^(l+1): [B, d_(l+1)]
      const WT = transpose(this.weights[l]);             // [d_(l+1), d_l]
      const a = matMul(XAbove, WT);                      // [B, d_l]
      const xHat = mapMatrix(a, this.activation.fn);     // [B, d_l]
      const e = sub(inputsLatents[l], xHat);             // X^(l) - Xhat^(l)
      const h = hadamard(e, mapMatrix(a, this.activation.deriv));

      A.push(a);
      XHat.push(xHat);
      E.push(e);
      H.push(h);
    }

    // Readout:
    // Yhat = X^(L) Wout^T
    const YHat = matMul(inputsLatents[this.L], transpose(this.wOut)); // [B, d_out]
    const ESup = sub(YHat, targets);                                  // [B, d_out]

    // Top latent error used during inference:
    // E^(L) = ESup * Wout
    const ETop = matMul(ESup, this.wOut);                             // [B, d_L]

    return { A, XHat, E, H, YHat, ESup, ETop };
  }

  /**
   * One training step on a batch.
   *
   * xBatch: [B, d0]
   * yBatch: [B, outputDim] one-hot targets
   */
  trainStep(xBatch: Matrix, yBatch: Matrix): TrainStepResult {
    const batchSize = xBatch.length;
    if (xBatch[0].length !== this.dims[0]) {
      throw new Error(`Expected input dim ${this.dims[0]}, got ${xBatch[0].length}`);
    }
    if (yBatch[0].length !== this.outputDim) {
      throw new Error(`Expected target dim ${this.outputDim}, got ${yBatch[0].length}`);
    }

    // inputsLatents = [X^(0), X^(1), ..., X^(L)]
    // X^(0) is clamped to the data; other X^(l) are inferred.
    const latents = this.initLatents(batchSize);
    const inputsLatents: Matrix[] = [cloneMatrix(xBatch), ...latents];

    /* ------------------------------ Inference loop ------------------------------ */
    for (let t = 0; t < this.tInfer; t++) {
      // IMPORTANT:
      // We first compute a full "snapshot" of all current errors and predictions,
      // then update every latent using that snapshot. This matches the paper's
      // synchronous inference update.  [oai_citation:2‡2506.06332v1.pdf](sediment://file_000000007f2471f5b1b0c437f64f4ed2)
      const cache = this.computeForwardQuantities(inputsLatents, yBatch);

      // Extend E with the top latent error E^(L) = ESup * Wout.
      const extendedE = [...cache.E, cache.ETop];

      // Update each latent X^(l), l = 1..L:
      // Gx^(l) = E^(l) - H^(l-1) W^(l-1)
      // X^(l)  = X^(l) - etaInfer * Gx^(l)
      for (let l = 1; l <= this.L; l++) {
        const feedback = matMul(cache.H[l - 1], this.weights[l - 1]); // [B, d_l]
        const gradX = sub(extendedE[l], feedback);                    // [B, d_l]
        inputsLatents[l] = sub(inputsLatents[l], scale(gradX, this.etaInfer));
      }
    }

    /* ------------------------------- Learning loop ------------------------------ */
    for (let t = 0; t < this.tLearn; t++) {
      // Recompute errors because weights change inside the learning loop.
      const cache = this.computeForwardQuantities(inputsLatents, yBatch);

      // Update generative weights:
      // Gw^(l) = -(1/B) H^(l)^T X^(l+1)
      // W^(l)  = W^(l) - etaLearn * Gw^(l)
      //
      // Since Gw has a leading minus sign, this is equivalent to adding
      // etaLearn * (1/B) H^(l)^T X^(l+1).
      for (let l = 0; l < this.L; l++) {
        const hT = transpose(cache.H[l]);                   // [d_l, B]
        const xAbove = inputsLatents[l + 1];               // [B, d_(l+1)]
        const gradW = scale(matMul(hT, xAbove), -1 / batchSize);
        this.weights[l] = sub(this.weights[l], scale(gradW, this.etaLearn));
      }

      // Update readout weights:
      // Gwout = (1/B) ESup^T X^(L)
      // Wout  = Wout - etaLearn * Gwout
      const esupT = transpose(cache.ESup);                 // [d_out, B]
      const xTop = inputsLatents[this.L];                  // [B, d_L]
      const gradWOut = scale(matMul(esupT, xTop), 1 / batchSize);
      this.wOut = sub(this.wOut, scale(gradWOut, this.etaLearn));
    }

    // Final logits after inference + learning.
    const finalCache = this.computeForwardQuantities(inputsLatents, yBatch);

    // A simple diagnostic loss:
    // mean of 0.5 * ||Esup||^2 over all entries
    const loss = meanSquaredHalf(finalCache.ESup);

    return {
      logits: finalCache.YHat,
      loss,
      latents: inputsLatents.slice(1),
    };
  }

  /**
   * Pure inference / prediction with frozen weights.
   * This matches the testing setup: randomly initialize latents, run inference,
   * then read out Yhat from the top latent.  [oai_citation:3‡2506.06332v1.pdf](sediment://file_000000007f2471f5b1b0c437f64f4ed2)
   */
  predict(xBatch: Matrix): Matrix {
    const batchSize = xBatch.length;
    const dummyTargets = zeros(batchSize, this.outputDim); // needed only for cache shape
    const latents = this.initLatents(batchSize);
    const inputsLatents: Matrix[] = [cloneMatrix(xBatch), ...latents];

    for (let t = 0; t < this.tInfer; t++) {
      const cache = this.computeForwardQuantities(inputsLatents, dummyTargets);

      // During pure prediction there is no target. A simple choice is to suppress
      // the supervised contribution by setting E^(L)=0.
      const zeroTop = zeros(batchSize, this.dims[this.L]);
      const extendedE = [...cache.E, zeroTop];

      for (let l = 1; l <= this.L; l++) {
        const feedback = matMul(cache.H[l - 1], this.weights[l - 1]);
        const gradX = sub(extendedE[l], feedback);
        inputsLatents[l] = sub(inputsLatents[l], scale(gradX, this.etaInfer));
      }
    }

    const yHat = matMul(inputsLatents[this.L], transpose(this.wOut));
    return yHat;
  }
}

/* -------------------------------------------------------------------------- */
/*                              Example usage                                 */
/* -------------------------------------------------------------------------- */

// Tiny toy dataset:
// inputs are 2D, labels are binary.
const x: Matrix = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

const labels = [0, 1, 1, 0]; // XOR targets
const y = oneHot(labels, 2);

// A tiny PCN: input d0=2, latent dims [6, 4], top latent dL=4, output dim=2
const pcn = new PredictiveCodingNetwork({
  dims: [2, 6, 4],
  outputDim: 2,
  etaInfer: 0.05,
  etaLearn: 0.01,
  tInfer: 20,
  tLearn: 10,
  latentInitStd: 0.01,
});

// Train for a few epochs
for (let epoch = 0; epoch < 200; epoch++) {
  const result = pcn.trainStep(x, y);
  if ((epoch + 1) % 20 === 0) {
    console.log(`Epoch ${epoch + 1}, loss=${result.loss.toFixed(6)}`);
  }
}

// Predict
const logits = pcn.predict(x);
console.log("Logits:");
console.log(logits);

// Argmax helper for class prediction
function argmax(row: Vector): number {
  let bestIdx = 0;
  let bestVal = row[0];
  for (let i = 1; i < row.length; i++) {
    if (row[i] > bestVal) {
      bestVal = row[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

const preds = logits.map(argmax);
console.log("Predictions:", preds);
console.log("Labels     :", labels);