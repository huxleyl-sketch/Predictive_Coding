"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
/* -------------------------------------------------------------------------- */
/*                                Math helpers                                */
/* -------------------------------------------------------------------------- */
function zeros(rows, cols) {
    return Array.from({ length: rows }, function () { return Array(cols).fill(0); });
}
function cloneMatrix(m) {
    return m.map(function (row) { return __spreadArray([], row, true); });
}
function transpose(m) {
    var rows = m.length;
    var cols = m[0].length;
    var out = zeros(cols, rows);
    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < cols; j++) {
            out[j][i] = m[i][j];
        }
    }
    return out;
}
function matMul(a, b) {
    var aRows = a.length;
    var aCols = a[0].length;
    var bRows = b.length;
    var bCols = b[0].length;
    if (aCols !== bRows) {
        throw new Error("matMul shape mismatch: [".concat(aRows, ",").concat(aCols, "] x [").concat(bRows, ",").concat(bCols, "]"));
    }
    var out = zeros(aRows, bCols);
    for (var i = 0; i < aRows; i++) {
        for (var k = 0; k < aCols; k++) {
            var aik = a[i][k];
            for (var j = 0; j < bCols; j++) {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    return out;
}
function add(a, b) {
    var out = zeros(a.length, a[0].length);
    for (var i = 0; i < a.length; i++) {
        for (var j = 0; j < a[0].length; j++) {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    return out;
}
function sub(a, b) {
    var out = zeros(a.length, a[0].length);
    for (var i = 0; i < a.length; i++) {
        for (var j = 0; j < a[0].length; j++) {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    return out;
}
function scale(a, s) {
    var out = zeros(a.length, a[0].length);
    for (var i = 0; i < a.length; i++) {
        for (var j = 0; j < a[0].length; j++) {
            out[i][j] = a[i][j] * s;
        }
    }
    return out;
}
function hadamard(a, b) {
    var out = zeros(a.length, a[0].length);
    for (var i = 0; i < a.length; i++) {
        for (var j = 0; j < a[0].length; j++) {
            out[i][j] = a[i][j] * b[i][j];
        }
    }
    return out;
}
function mapMatrix(a, fn) {
    var out = zeros(a.length, a[0].length);
    for (var i = 0; i < a.length; i++) {
        for (var j = 0; j < a[0].length; j++) {
            out[i][j] = fn(a[i][j]);
        }
    }
    return out;
}
function meanSquaredHalf(a) {
    var sum = 0;
    var count = 0;
    for (var _i = 0, a_1 = a; _i < a_1.length; _i++) {
        var row = a_1[_i];
        for (var _a = 0, row_1 = row; _a < row_1.length; _a++) {
            var x_1 = row_1[_a];
            sum += x_1 * x_1;
            count++;
        }
    }
    return 0.5 * sum / Math.max(count, 1);
}
/* -------------------------------------------------------------------------- */
/*                         Random init / utility helpers                      */
/* -------------------------------------------------------------------------- */
function randn() {
    // Box-Muller transform
    var u = 0;
    var v = 0;
    while (u === 0)
        u = Math.random();
    while (v === 0)
        v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function randomMatrix(rows, cols, std) {
    if (std === void 0) { std = 1; }
    var out = zeros(rows, cols);
    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < cols; j++) {
            out[i][j] = randn() * std;
        }
    }
    return out;
}
function xavierUniform(rows, cols) {
    // For W shape [fanOut, fanIn]
    var limit = Math.sqrt(6 / (rows + cols));
    var out = zeros(rows, cols);
    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < cols; j++) {
            out[i][j] = (Math.random() * 2 - 1) * limit;
        }
    }
    return out;
}
function oneHot(labels, numClasses) {
    var out = zeros(labels.length, numClasses);
    for (var i = 0; i < labels.length; i++) {
        out[i][labels[i]] = 1;
    }
    return out;
}
/* -------------------------------------------------------------------------- */
/*                           Predictive Coding Network                        */
/* -------------------------------------------------------------------------- */
var PredictiveCodingNetwork = /** @class */ (function () {
    function PredictiveCodingNetwork(config) {
        var _a, _b;
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
        this.latentInitStd = (_a = config.latentInitStd) !== null && _a !== void 0 ? _a : 0.01;
        this.activation = (_b = config.activation) !== null && _b !== void 0 ? _b : {
            fn: function (x) { return Math.max(0, x); }, // ReLU
            deriv: function (x) { return (x > 0 ? 1 : 0); }, // d/dx ReLU
        };
        // Initialize generative weights W^(0)...W^(L-1)
        this.weights = [];
        for (var l = 0; l < this.L; l++) {
            var dLower = this.dims[l];
            var dUpper = this.dims[l + 1];
            this.weights.push(xavierUniform(dLower, dUpper));
        }
        // Initialize readout Wout
        this.wOut = xavierUniform(this.outputDim, this.dims[this.L]);
    }
    /**
     * Initialize latent states X^(1)...X^(L) with small Gaussian noise.
     * This follows the paper's setup: latents are freshly initialized per batch/sample.
     */
    PredictiveCodingNetwork.prototype.initLatents = function (batchSize) {
        var latents = [];
        for (var l = 1; l <= this.L; l++) {
            latents.push(randomMatrix(batchSize, this.dims[l], this.latentInitStd));
        }
        return latents;
    };
    /**
     * Compute the per-layer quantities:
     *   A^(l), Xhat^(l), E^(l), H^(l)
     * plus:
     *   Yhat, ESup, ETop
     *
     * inputsLatents is [X^(0), X^(1), ..., X^(L)]
     */
    PredictiveCodingNetwork.prototype.computeForwardQuantities = function (inputsLatents, targets) {
        var A = [];
        var XHat = [];
        var E = [];
        var H = [];
        // For each generative layer l = 0..L-1:
        // A^(l) = X^(l+1) W^(l)^T
        for (var l = 0; l < this.L; l++) {
            var XAbove = inputsLatents[l + 1]; // X^(l+1): [B, d_(l+1)]
            var WT = transpose(this.weights[l]); // [d_(l+1), d_l]
            var a = matMul(XAbove, WT); // [B, d_l]
            var xHat = mapMatrix(a, this.activation.fn); // [B, d_l]
            var e = sub(inputsLatents[l], xHat); // X^(l) - Xhat^(l)
            var h = hadamard(e, mapMatrix(a, this.activation.deriv));
            A.push(a);
            XHat.push(xHat);
            E.push(e);
            H.push(h);
        }
        // Readout:
        // Yhat = X^(L) Wout^T
        var YHat = matMul(inputsLatents[this.L], transpose(this.wOut)); // [B, d_out]
        var ESup = sub(YHat, targets); // [B, d_out]
        // Top latent error used during inference:
        // E^(L) = ESup * Wout
        var ETop = matMul(ESup, this.wOut); // [B, d_L]
        return { A: A, XHat: XHat, E: E, H: H, YHat: YHat, ESup: ESup, ETop: ETop };
    };
    /**
     * One training step on a batch.
     *
     * xBatch: [B, d0]
     * yBatch: [B, outputDim] one-hot targets
     */
    PredictiveCodingNetwork.prototype.trainStep = function (xBatch, yBatch) {
        var batchSize = xBatch.length;
        if (xBatch[0].length !== this.dims[0]) {
            throw new Error("Expected input dim ".concat(this.dims[0], ", got ").concat(xBatch[0].length));
        }
        if (yBatch[0].length !== this.outputDim) {
            throw new Error("Expected target dim ".concat(this.outputDim, ", got ").concat(yBatch[0].length));
        }
        // inputsLatents = [X^(0), X^(1), ..., X^(L)]
        // X^(0) is clamped to the data; other X^(l) are inferred.
        var latents = this.initLatents(batchSize);
        var inputsLatents = __spreadArray([cloneMatrix(xBatch)], latents, true);
        /* ------------------------------ Inference loop ------------------------------ */
        for (var t = 0; t < this.tInfer; t++) {
            // IMPORTANT:
            // We first compute a full "snapshot" of all current errors and predictions,
            // then update every latent using that snapshot. This matches the paper's
            // synchronous inference update.  [oai_citation:2‡2506.06332v1.pdf](sediment://file_000000007f2471f5b1b0c437f64f4ed2)
            var cache = this.computeForwardQuantities(inputsLatents, yBatch);
            // Extend E with the top latent error E^(L) = ESup * Wout.
            var extendedE = __spreadArray(__spreadArray([], cache.E, true), [cache.ETop], false);
            // Update each latent X^(l), l = 1..L:
            // Gx^(l) = E^(l) - H^(l-1) W^(l-1)
            // X^(l)  = X^(l) - etaInfer * Gx^(l)
            for (var l = 1; l <= this.L; l++) {
                var feedback = matMul(cache.H[l - 1], this.weights[l - 1]); // [B, d_l]
                var gradX = sub(extendedE[l], feedback); // [B, d_l]
                inputsLatents[l] = sub(inputsLatents[l], scale(gradX, this.etaInfer));
            }
        }
        /* ------------------------------- Learning loop ------------------------------ */
        for (var t = 0; t < this.tLearn; t++) {
            // Recompute errors because weights change inside the learning loop.
            var cache = this.computeForwardQuantities(inputsLatents, yBatch);
            // Update generative weights:
            // Gw^(l) = -(1/B) H^(l)^T X^(l+1)
            // W^(l)  = W^(l) - etaLearn * Gw^(l)
            //
            // Since Gw has a leading minus sign, this is equivalent to adding
            // etaLearn * (1/B) H^(l)^T X^(l+1).
            for (var l = 0; l < this.L; l++) {
                var hT = transpose(cache.H[l]); // [d_l, B]
                var xAbove = inputsLatents[l + 1]; // [B, d_(l+1)]
                var gradW = scale(matMul(hT, xAbove), -1 / batchSize);
                this.weights[l] = sub(this.weights[l], scale(gradW, this.etaLearn));
            }
            // Update readout weights:
            // Gwout = (1/B) ESup^T X^(L)
            // Wout  = Wout - etaLearn * Gwout
            var esupT = transpose(cache.ESup); // [d_out, B]
            var xTop = inputsLatents[this.L]; // [B, d_L]
            var gradWOut = scale(matMul(esupT, xTop), 1 / batchSize);
            this.wOut = sub(this.wOut, scale(gradWOut, this.etaLearn));
        }
        // Final logits after inference + learning.
        var finalCache = this.computeForwardQuantities(inputsLatents, yBatch);
        // A simple diagnostic loss:
        // mean of 0.5 * ||Esup||^2 over all entries
        var loss = meanSquaredHalf(finalCache.ESup);
        return {
            logits: finalCache.YHat,
            loss: loss,
            latents: inputsLatents.slice(1),
        };
    };
    /**
     * Pure inference / prediction with frozen weights.
     * This matches the testing setup: randomly initialize latents, run inference,
     * then read out Yhat from the top latent.  [oai_citation:3‡2506.06332v1.pdf](sediment://file_000000007f2471f5b1b0c437f64f4ed2)
     */
    PredictiveCodingNetwork.prototype.predict = function (xBatch) {
        var batchSize = xBatch.length;
        var dummyTargets = zeros(batchSize, this.outputDim); // needed only for cache shape
        var latents = this.initLatents(batchSize);
        var inputsLatents = __spreadArray([cloneMatrix(xBatch)], latents, true);
        for (var t = 0; t < this.tInfer; t++) {
            var cache = this.computeForwardQuantities(inputsLatents, dummyTargets);
            // During pure prediction there is no target. A simple choice is to suppress
            // the supervised contribution by setting E^(L)=0.
            var zeroTop = zeros(batchSize, this.dims[this.L]);
            var extendedE = __spreadArray(__spreadArray([], cache.E, true), [zeroTop], false);
            for (var l = 1; l <= this.L; l++) {
                var feedback = matMul(cache.H[l - 1], this.weights[l - 1]);
                var gradX = sub(extendedE[l], feedback);
                inputsLatents[l] = sub(inputsLatents[l], scale(gradX, this.etaInfer));
            }
        }
        var yHat = matMul(inputsLatents[this.L], transpose(this.wOut));
        return yHat;
    };
    return PredictiveCodingNetwork;
}());
/* -------------------------------------------------------------------------- */
/*                              Example usage                                 */
/* -------------------------------------------------------------------------- */
// Tiny toy dataset:
// inputs are 2D, labels are binary.
var x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
var labels = [0, 1, 1, 0]; // XOR targets
var y = oneHot(labels, 2);
// A tiny PCN: input d0=2, latent dims [6, 4], top latent dL=4, output dim=2
var pcn = new PredictiveCodingNetwork({
    dims: [2, 6, 4],
    outputDim: 2,
    etaInfer: 0.05,
    etaLearn: 0.01,
    tInfer: 20,
    tLearn: 10,
    latentInitStd: 0.01,
});
// Train for a few epochs
for (var epoch = 0; epoch < 200; epoch++) {
    var result = pcn.trainStep(x, y);
    if ((epoch + 1) % 20 === 0) {
        console.log("Epoch ".concat(epoch + 1, ", loss=").concat(result.loss.toFixed(6)));
    }
}
// Predict
var logits = pcn.predict(x);
console.log("Logits:");
console.log(logits);
// Argmax helper for class prediction
function argmax(row) {
    var bestIdx = 0;
    var bestVal = row[0];
    for (var i = 1; i < row.length; i++) {
        if (row[i] > bestVal) {
            bestVal = row[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}
var preds = logits.map(argmax);
console.log("Predictions:", preds);
console.log("Labels     :", labels);
