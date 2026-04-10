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
    private panels: {
        title: string;
        matrix: [[number, number], [number, number]];
        total: number;
        accuracy: number;
        positiveClassName: string;
    }[] = [];

    constructor(canvas: HTMLCanvasElement | null) {
        this.canvas = canvas;
        this.ctx = canvas?.getContext("2d") ?? null;
    }

    setFromClasses(
        predicted: ArrayLike<number>,
        target: ArrayLike<number>,
        labels: string[] = ["Positive", "Abstain"],
    ) {
        const positiveClassName = labels[0] ?? "Positive";
        this.panels = [this.buildBinaryPanel("Confusion Matrix", predicted, target, positiveClassName)];
        this.render();
    }

    setDualPcnBinary(
        upPredicted: ArrayLike<number>,
        upTarget: ArrayLike<number>,
        downPredicted: ArrayLike<number>,
        downTarget: ArrayLike<number>,
        options?: {
            upPositiveClassName?: string;
            downPositiveClassName?: string;
        },
    ) {
        const upPositiveClassName = options?.upPositiveClassName ?? "Up/Flat";
        const downPositiveClassName = options?.downPositiveClassName ?? "Down";

        this.panels = [
            this.buildBinaryPanel("UP PCN", upPredicted, upTarget, upPositiveClassName),
            this.buildBinaryPanel("DOWN PCN", downPredicted, downTarget, downPositiveClassName),
        ];
        this.render();
    }

    render() {
        if (!this.canvas || !this.ctx) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, 0, width, height);
        ctx.font = "12px IBM Plex Sans, sans-serif";

        ctx.fillStyle = "#293241";
        ctx.fillText("Interpretation: PCN Confusion Matrices", 10, 16);
        ctx.fillText("Rows = Actual, Columns = Predicted (Positive vs Abstain)", 10, 32);

        if (this.panels.length === 0) {
            ctx.fillStyle = "#6f7784";
            ctx.fillText("No prediction data yet.", 10, 52);
            return;
        }

        const panelTop = 38;
        const panelBottom = 8;
        const panelHeight = height - panelTop - panelBottom;
        const gap = 12;
        const sidePad = 10;

        if (this.panels.length === 1) {
            this.drawBinaryPanel(this.panels[0]!, sidePad, panelTop, width - sidePad * 2, panelHeight);
            return;
        }

        const panelWidth = (width - sidePad * 2 - gap) / 2;
        this.drawBinaryPanel(this.panels[0]!, sidePad, panelTop, panelWidth, panelHeight);
        this.drawBinaryPanel(this.panels[1]!, sidePad + panelWidth + gap, panelTop, panelWidth, panelHeight);
    }

    private buildBinaryPanel(
        title: string,
        predicted: ArrayLike<number>,
        target: ArrayLike<number>,
        positiveClassName: string,
    ) {
        const n = Math.min(predicted.length, target.length);
        const matrix: [[number, number], [number, number]] = [
            [0, 0],
            [0, 0],
        ];

        let total = 0;
        let matches = 0;

        for (let i = 0; i < n; i++) {
            const p = Number(predicted[i]);
            const t = Number(target[i]);
            if (!Number.isFinite(p) || !Number.isFinite(t)) continue;
            const pBin: 0 | 1 = p === 0 ? 0 : 1;
            const tBin: 0 | 1 = t === 0 ? 0 : 1;
            matrix[tBin][pBin] += 1;
            total++;
            if (pBin === tBin) matches++;
        }

        return {
            title,
            matrix,
            total,
            accuracy: total > 0 ? matches / total : 0,
            positiveClassName,
        };
    }

    private drawBinaryPanel(
        panel: {
            title: string;
            matrix: [[number, number], [number, number]];
            total: number;
            accuracy: number;
            positiveClassName: string;
        },
        x: number,
        y: number,
        w: number,
        h: number,
    ) {
        if (!this.ctx) return;
        const ctx = this.ctx;

        const labelPadLeft = 70;
        const labelPadTop = 20;
        const labelPadBottom = 28;
        const gridTop = y + 24 + labelPadTop;
        const gridAvailableH = h - 24 - labelPadTop - labelPadBottom;
        const gridAvailableW = w - labelPadLeft - 12;
        const gridSize = Math.max(20, Math.min(gridAvailableW, gridAvailableH));
        const cellSize = gridSize / 2;
        const gridX = x + labelPadLeft;
        const gridY = gridTop;

        ctx.fillStyle = "#1f2937";
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(`${panel.title} (Positive=${panel.positiveClassName})`, x + 4, y + 14);

        ctx.fillStyle = "#374151";
        ctx.textAlign = "center";
        ctx.fillText("Predicted", gridX + gridSize / 2, y + 26);

        ctx.save();
        ctx.translate(x + 16, gridY + gridSize / 2 + 8);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Actual", 0, 0);
        ctx.restore();

        const maxCell = Math.max(
            panel.matrix[0][0],
            panel.matrix[0][1],
            panel.matrix[1][0],
            panel.matrix[1][1],
            1,
        );

        for (let r = 0; r < 2; r++) {
            for (let c = 0; c < 2; c++) {
                const count = panel.matrix[r]![c]!;
                const strength = count / maxCell;
                const cx = gridX + c * cellSize;
                const cy = gridY + r * cellSize;

                ctx.fillStyle = `rgba(33, 158, 188, ${0.12 + 0.78 * strength})`;
                ctx.fillRect(cx, cy, cellSize, cellSize);
                ctx.strokeStyle = "#ffffff";
                ctx.strokeRect(cx, cy, cellSize, cellSize);

                ctx.fillStyle = "#0f172a";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(String(count), cx + cellSize / 2, cy + cellSize / 2);
            }
        }

        const classLabels = ["Positive", "Abstain"];
        ctx.fillStyle = "#374151";
        ctx.textBaseline = "alphabetic";
        ctx.textAlign = "center";
        for (let c = 0; c < 2; c++) {
            ctx.fillText(classLabels[c]!, gridX + c * cellSize + cellSize / 2, gridY - 6);
        }

        ctx.textAlign = "right";
        for (let r = 0; r < 2; r++) {
            ctx.fillText(classLabels[r]!, gridX - 6, gridY + r * cellSize + cellSize / 2 + 4);
        }

        ctx.textAlign = "left";
        ctx.fillStyle = "#1f2937";
        ctx.fillText(`Accuracy: ${(panel.accuracy * 100).toFixed(2)}%`, x + 4, y + h - 8);
        ctx.fillText(`Samples: ${panel.total}`, x + w * 0.52, y + h - 8);
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
