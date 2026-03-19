import * as tf from '@tensorflow/tfjs';

// Ensure a backend is registered before any eager ops run.
import '@tensorflow/tfjs-backend-webgl';
await tf.setBackend('webgl');
await tf.ready();

type PCNconfig = {
    epochs: number,
    T_infer: number,
    eta_infer: number,
    T_learn: number,
    eta_learn: number,
    data: {
        x: tf.Tensor2D,
        y: tf.Tensor2D,
    }[]
}

class PCN {
    
    /** Latents */
    X: tf.Tensor2D[];
    /** Weights */
    W: tf.Tensor2D[];
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

    /**
     * @param dims Dimensions
     */
    constructor( dims: number[]){
        this.f = (m: tf.Tensor2D) => m.relu();
        this.df = (m: tf.Tensor2D) => m.max(0);

        this.H = [];

        this.E = [];

        this.X = [];
        this.W = [];

        this.D = dims;
        this.L = dims.length - 2;
    }

    Train(input: PCNconfig = {
        epochs: 2000,
        T_infer: 50,
        eta_infer: 0.05,
        T_learn: 500,
        eta_learn: 0.005,
        data:[{
            x: tf.tensor2d([]),
            y: tf.tensor2d([]),
        }],
    }){
        const T_infer = input.T_infer;
        const eta_infer = input.eta_infer;
        const T_learn = input.T_learn;
        const eta_learn = input.eta_learn;

        for(let epoch = 0; epoch < input.epochs; epoch++){
        for(let batch of input.data){
            /** Batch Size */
            const B = batch.x.shape[0]!;
            this.H[0] = zeros(B,this.D[0]!);
            this.H[this.L + 1] = zeros(B,this.D[this.L + 1]!)

            /** Initialise Values */
            for( let l = 0; l < this.L; l++){
                this.W[l] = xavierUniform( this.D[l]!, this.D[l + 1]! );
                /** Small Random Values */
                this.X[l+1] = xavierUniform( B, this.D[l+1]! );
            }
            
            this.W[this.L + 1] = xavierUniform( this.D[this.L + 1]!, this.D[this.L]! )

            /** Input Batch Fixing */
            this.X[0] = batch.x;
            const Y = batch.y;

            /** Inference Update Loop */
            for(let t = 1; t <= T_infer; t++){
                for(let l = 0; l < this.L; l++){
                    /** Pre Activation Predicted Latent Values */
                    const A_l = tf.matMul(this.X[l+1], this.W[l].transpose()) satisfies tf.Tensor2D;
                    /** Predicted Latent Values */
                    const Xhat_l = this.f(A_l);
                    this.E[l] = tf.sub(this.X[l], Xhat_l);
                    this.H[l] = tf.mul(this.E[l], this.df(A_l))
                }

                let Yhat = tf.matMul(this.X[this.L], this.W[this.L+1].transpose());
        
                let E_sup = tf.sub(Yhat, Y);
                this.E[this.L] = tf.matMul(E_sup, this.W[this.L+1]);

                for(let l = 1; l <= this.L; l++){
                    /** Gradients of Latents */
                    const G_xl = tf.sub(this.E[l], tf.matMul(this.H[l-1], this.W[l-1]));
                    this.X[l] = tf.sub(this.X[l], tf.mul(eta_infer, G_xl));
                }
            }
            /** Weight Update Loop */
            for(let t = 1; t <= T_learn; t++){
                
                for(let l = 0; l < this.L; l++){
                    const A_l = tf.matMul(this.X[l+1],this.W[l].transpose()) satisfies tf.Tensor2D;
                    const Xhat_l = this.f(A_l);
                    this.E[l] = tf.sub(this.X[l], Xhat_l);
                    this.H[l] = tf.mul(this.E[l],this.df(A_l))
                    /** Average Gradients of Weights */
                    const G_wl = tf.mul(tf.scalar(-1/B), tf.matMul(this.H[l].transpose(), this.X[l+1]))
                    this.W[l] = tf.sub(this.W[l], tf.mul(eta_learn, G_wl));
                }
                let Yhat = tf.matMul(this.X[this.L], this.W[this.L+1].transpose());
                let E_sup = tf.sub(Yhat, Y);
                /** Average Gradients of output weight */
                const G_w_out = tf.mul(1/B,tf.matMul(E_sup.transpose(), this.X[this.L]));
                this.W[this.L+1] = tf.sub(this.W[this.L+1], tf.mul(eta_learn, G_w_out)); 
            }
        }}
        let debug = document.getElementById("debug");
        if(debug) debug.innerText = "Complete";
    }


    

    Generate(xBatch: tf.Tensor2D, T_infer: number, eta_infer: number){
        this.X[0] = xBatch;
        for( let l = 1; l <= this.L; l++){
            /** Small Random Values */
            this.X[l] = xavierUniform( xBatch.shape[0], this.D[l]! );
        }
        /** Inference Update Loop */
        for(let t = 1; t <= T_infer; t++){
            for(let l = 0; l < this.L; l++){
                /** Pre Activation Predicted Latent Values */
                const A_l = tf.matMul(this.X[l+1], this.W[l].transpose()) satisfies tf.Tensor2D;
                /** Predicted Latent Values */
                const Xhat_l = this.f(A_l);
                this.E[l] = tf.sub(this.X[l], Xhat_l);
                this.H[l] = tf.mul(this.E[l], this.df(A_l))
            }            
            this.E[this.L] = zeros(xBatch.shape[0]!,this.D[this.L]!)

            for(let l = 1; l <= this.L; l++){
                /** Gradients of Latents */
                const G_xl = tf.sub(this.E[l], tf.matMul(this.H[l-1], this.W[l-1]));
                this.X[l] = tf.sub(this.X[l], tf.mul(eta_infer, G_xl));
            }
        }
        return tf.matMul(this.X[this.L], this.W[this.L + 1].transpose()) satisfies tf.Tensor2D;
    }
}

let x = tf.tensor2d([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])
let y = tf.tensor2d([
    [1,0],
    [0,1],
    [0,1],
    [1,0],
])

let pcnSetup: PCNconfig = {
    epochs: 100,
    T_infer: 50,
    eta_infer: 0.05,
    T_learn: 10,
    eta_learn: 0.005,
    data: [{ x, y, }],
};

let pcn = new PCN([2,6,4,2]);
pcn.Train(pcnSetup);
document.getElementById("retry")?.addEventListener('click', () => {
    let result = pcn.Generate(x, 50, 0.05);
    
    console.log("Result:", argMax(result))
    console.log("Target:", argMax(y))
});
function argMax(mat: tf.Tensor2D): number[]{
    let arr = mat.arraySync();
    return arr.map(row => {
        return row[0] > row[1] ? 0 : 1;
    })
}

function zeros(rows: number, cols: number): tf.Tensor2D{
    return tf.tensor2d(Array(cols).fill(Array(rows).fill(0)));
}
function xavierUniform(rows: number, cols: number): tf.Tensor2D {
    const limit = Math.sqrt(6 / (rows + cols));
    const out: number[][] = [];

    for (let i = 0; i < rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < cols; j++) {
            row.push((Math.random() * 2 - 1) * limit);
        }
        out.push(row);
    }

    return tf.tensor2d(out);
}
