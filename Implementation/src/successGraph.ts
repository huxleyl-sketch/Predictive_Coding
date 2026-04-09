import * as tf from '@tensorflow/tfjs';

export class SuccessGraph {
    private canvas: HTMLCanvasElement | null;
    private ctx: CanvasRenderingContext2D | null;
    private history: number[] = [];
    private title: string;
    private xAxisLabel: string;
    private baselineLines: { label: string; value: number; color: string }[] = [];

    constructor(
        canvas: HTMLCanvasElement | null,
        options?: {
            title?: string;
            xAxisLabel?: string;
        },
    ) {
        this.canvas = canvas;
        this.ctx = canvas?.getContext("2d") ?? null;
        this.title = options?.title ?? "Success";
        this.xAxisLabel = options?.xAxisLabel ?? "Step";
    }

    setHistory(values: number[]) {
        this.history = values.map((v) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0)));
        this.render();
    }

    setBaselines(lines: { label: string; value: number; color?: string }[]) {
        this.baselineLines = lines
            .filter((line) => Number.isFinite(line.value))
            .map((line, idx) => ({
                label: line.label,
                value: Math.max(0, Math.min(1, line.value)),
                color: line.color ?? (idx % 2 === 0 ? "#f59e0b" : "#6366f1"),
            }));
        this.render();
    }

    render() {
        if (!this.canvas || !this.ctx) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const left = 48;
        const right = 16;
        const top = 18;
        const bottom = 34;
        const chartW = width - left - right;
        const chartH = height - top - bottom;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, 0, width, height);

        ctx.font = "12px IBM Plex Sans, sans-serif";
        ctx.strokeStyle = "#d6dce8";
        ctx.fillStyle = "#5a6473";
        ctx.lineWidth = 1;

        for (let i = 0; i <= 4; i++) {
            const value = i / 4;
            const y = top + (1 - value) * chartH;
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(width - right, y);
            ctx.stroke();
            ctx.fillText(`${Math.round(value * 100)}%`, 8, y + 4);
        }

        ctx.strokeStyle = "#7c8796";
        ctx.beginPath();
        ctx.moveTo(left, top);
        ctx.lineTo(left, height - bottom);
        ctx.lineTo(width - right, height - bottom);
        ctx.stroke();

        ctx.fillStyle = "#293241";
        ctx.fillText(this.title, 8, 12);
        ctx.fillText(this.xAxisLabel, width - 72, height - 10);

        for (let i = 0; i < this.baselineLines.length; i++) {
            const line = this.baselineLines[i]!;
            const y = top + (1 - line.value) * chartH;
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = line.color;
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(width - right, y);
            ctx.stroke();
            ctx.restore();

            ctx.fillStyle = line.color;
            ctx.fillText(`${line.label} ${(line.value * 100).toFixed(1)}%`, left + 6, y - 4);
        }

        if (this.history.length === 0) {
            ctx.fillStyle = "#6f7784";
            ctx.fillText("No accuracy history yet.", left + 10, top + 18);
            return;
        }

        const toX = (idx: number) =>
            this.history.length === 1
                ? left + chartW / 2
                : left + (idx / (this.history.length - 1)) * chartW;
        const toY = (v: number) => top + (1 - v) * chartH;

        ctx.strokeStyle = "#118ab2";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < this.history.length; i++) {
            const x = toX(i);
            const y = toY(this.history[i]!);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = "#ef476f";
        for (let i = 0; i < this.history.length; i++) {
            const x = toX(i);
            const y = toY(this.history[i]!);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        const latest = this.history[this.history.length - 1]!;
        ctx.fillStyle = "#1f2937";
        ctx.fillText(`Latest: ${(latest * 100).toFixed(2)}%`, width - 156, 14);
    }

    record(score: number): number {
        const clamped = Math.max(0, Math.min(1, score));
        this.history.push(clamped);
        this.render();
        return clamped;
    }
}

export class ErrorGraph {
    private canvas: HTMLCanvasElement | null;
    private ctx: CanvasRenderingContext2D | null;
    private history: number[] = [];

    constructor(canvas: HTMLCanvasElement | null) {
        this.canvas = canvas;
        this.ctx = canvas?.getContext("2d") ?? null;
    }

    setHistory(values: number[]) {
        this.history = values.map((v) => (Number.isFinite(v) ? Math.max(0, v) : 0));
        this.render();
    }

    render() {
        if (!this.canvas || !this.ctx) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const left = 60;
        const right = 16;
        const top = 18;
        const bottom = 34;
        const chartW = width - left - right;
        const chartH = height - top - bottom;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, 0, width, height);

        ctx.font = "12px IBM Plex Sans, sans-serif";
        ctx.strokeStyle = "#d6dce8";
        ctx.fillStyle = "#5a6473";
        ctx.lineWidth = 1;

        const maxY = this.history.length > 0 ? Math.max(...this.history, 1e-6) : 1;

        for (let i = 0; i <= 4; i++) {
            const frac = i / 4;
            const y = top + (1 - frac) * chartH;
            const label = (frac * maxY).toFixed(4);
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(width - right, y);
            ctx.stroke();
            ctx.fillText(label, 8, y + 4);
        }

        ctx.strokeStyle = "#7c8796";
        ctx.beginPath();
        ctx.moveTo(left, top);
        ctx.lineTo(left, height - bottom);
        ctx.lineTo(width - right, height - bottom);
        ctx.stroke();

        ctx.fillStyle = "#293241";
        ctx.fillText("MSE Error", 8, 12);
        ctx.fillText("Epoch", width - 56, height - 10);

        if (this.history.length === 0) {
            ctx.fillStyle = "#6f7784";
            ctx.fillText("No error data yet.", left + 10, top + 18);
            return;
        }

        const toX = (idx: number) =>
            this.history.length === 1
                ? left + chartW / 2
                : left + (idx / (this.history.length - 1)) * chartW;
        const toY = (value: number) => top + (1 - value / maxY) * chartH;

        ctx.strokeStyle = "#ff7f11";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < this.history.length; i++) {
            const x = toX(i);
            const y = toY(this.history[i]!);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = "#c1121f";
        for (let i = 0; i < this.history.length; i++) {
            const x = toX(i);
            const y = toY(this.history[i]!);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        const latest = this.history[this.history.length - 1]!;
        ctx.fillStyle = "#1f2937";
        ctx.fillText(`Latest: ${latest.toFixed(6)}`, width - 176, 14);
    }
}

export class InterpretationGraph {
    private canvas: HTMLCanvasElement | null;
    private ctx: CanvasRenderingContext2D | null;
    private matrix: number[][] = [];
    private total = 0;
    private accuracy = 0;
    private classLabels: string[] = [];

    constructor(canvas: HTMLCanvasElement | null) {
        this.canvas = canvas;
        this.ctx = canvas?.getContext("2d") ?? null;
    }

    setFromClasses(
        predicted: ArrayLike<number>,
        target: ArrayLike<number>,
        labels: string[] = ["Up/Flat", "Down"],
    ) {
        const n = Math.min(predicted.length, target.length);
        if (n === 0) {
            this.matrix = [];
            this.total = 0;
            this.accuracy = 0;
            this.classLabels = labels;
            this.render();
            return;
        }

        let maxClass = 1;
        for (let i = 0; i < n; i++) {
            maxClass = Math.max(maxClass, Number(predicted[i] ?? 0), Number(target[i] ?? 0));
        }
        const classCount = Math.max(2, maxClass + 1);
        this.matrix = Array.from({ length: classCount }, () => new Array(classCount).fill(0));
        this.classLabels = labels.length >= classCount
            ? labels.slice(0, classCount)
            : Array.from({ length: classCount }, (_, i) => `Class ${i}`);

        let matches = 0;
        this.total = 0;

        for (let i = 0; i < n; i++) {
            const p = Number(predicted[i]);
            const t = Number(target[i]);
            if (!Number.isInteger(p) || !Number.isInteger(t)) continue;
            if (t < 0 || t >= classCount || p < 0 || p >= classCount) continue;
            this.matrix[t]![p]! += 1;
            this.total += 1;
            if (p === t) matches += 1;
        }

        this.accuracy = this.total > 0 ? matches / this.total : 0;
        this.render();
    }

    render() {
        if (!this.canvas || !this.ctx) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const left = 130;
        const right = 20;
        const top = 52;
        const bottom = 36;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, 0, width, height);
        ctx.font = "12px IBM Plex Sans, sans-serif";

        ctx.fillStyle = "#293241";
        ctx.fillText("Interpretation: Confusion Matrix", 10, 16);
        ctx.fillText("Rows = Actual, Columns = Predicted", 10, 32);

        if (this.total === 0 || this.matrix.length === 0) {
            ctx.fillStyle = "#6f7784";
            ctx.fillText("No prediction data yet.", 10, 52);
            return;
        }

        const classCount = this.matrix.length;
        const chartW = width - left - right;
        const chartH = height - top - bottom;
        const cellSize = Math.min(chartW / classCount, chartH / classCount);
        const gridW = cellSize * classCount;
        const gridH = cellSize * classCount;
        const gridX = left;
        const gridY = top;
        const maxCell = Math.max(...this.matrix.flat(), 1);

        ctx.fillStyle = "#374151";
        ctx.fillText("Predicted", gridX + gridW / 2 - 28, top - 26);
        ctx.save();
        ctx.translate(18, gridY + gridH / 2 + 24);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Actual", 0, 0);
        ctx.restore();

        for (let r = 0; r < classCount; r++) {
            for (let c = 0; c < classCount; c++) {
                const count = this.matrix[r]![c]!;
                const strength = count / maxCell;
                const x = gridX + c * cellSize;
                const y = gridY + r * cellSize;

                ctx.fillStyle = `rgba(33, 158, 188, ${0.12 + 0.78 * strength})`;
                ctx.fillRect(x, y, cellSize, cellSize);
                ctx.strokeStyle = "#ffffff";
                ctx.strokeRect(x, y, cellSize, cellSize);

                ctx.fillStyle = "#0f172a";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(String(count), x + cellSize / 2, y + cellSize / 2);
            }
        }

        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = "#374151";
        for (let c = 0; c < classCount; c++) {
            const x = gridX + c * cellSize + cellSize / 2;
            ctx.fillText(this.classLabels[c]!, x, gridY - 8);
        }

        ctx.textAlign = "right";
        for (let r = 0; r < classCount; r++) {
            const y = gridY + r * cellSize + cellSize / 2 + 4;
            ctx.fillText(this.classLabels[r]!, gridX - 8, y);
        }

        ctx.textAlign = "left";
        ctx.fillStyle = "#1f2937";
        ctx.fillText(`Accuracy: ${(this.accuracy * 100).toFixed(2)}%`, 10, height - 12);
        ctx.fillText(`Samples: ${this.total}`, 170, height - 12);
    }
}

export function computeSuccess(predictions: tf.Tensor2D, target: tf.Tensor2D): number {
    const predictedClass = tf.argMax(predictions, 1);
    const targetClass = tf.argMax(target, 1);
    const accuracy = tf.mean(tf.cast(tf.equal(predictedClass, targetClass), "float32"));
    const score = accuracy.dataSync()[0] ?? 0;
    predictedClass.dispose();
    targetClass.dispose();
    accuracy.dispose();
    return score;
}
