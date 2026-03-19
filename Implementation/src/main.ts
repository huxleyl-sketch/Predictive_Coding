import * as tf from '@tensorflow/tfjs';

// Ensure a backend is registered before any eager ops run.
import '@tensorflow/tfjs-backend-webgl';
await tf.setBackend('webgl');
await tf.ready();

let debug = document.getElementById("debug");

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

    private keepAssign(arr: tf.Tensor2D[], index: number, value: tf.Tensor2D) {
        const prev = arr[index];
        arr[index] = tf.keep(value) as tf.Tensor2D;
        prev?.dispose();
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
        console.time("Training");
        const T_infer = input.T_infer;
        const eta_infer = input.eta_infer;
        const T_learn = input.T_learn;
        const eta_learn = input.eta_learn;
        for( let l = 0; l < this.L; l++){
            this.W[l] = tf.variable(xavierUniform( this.D[l]!, this.D[l + 1]! ));
        }   
        this.W[this.L + 1] = tf.variable(xavierUniform( this.D[this.L + 1]!, this.D[this.L]! ))
        
        for(let epoch = 0; epoch < input.epochs; epoch++){
        for(let batch of input.data){
            /** Batch Size */
            const B = batch.x.shape[0]!;
            const negInvB = tf.scalar(-1 / B);
            const invB = tf.scalar(1 / B);
            this.H[0] = zeros(B,this.D[0]!);
            this.H[this.L + 1] = zeros(B,this.D[this.L + 1]!)

            /** Initialise Values */
            for( let l = 1; l <= this.L; l++){
                /** Small Random Values */
                this.X[l] = xavierUniform( B, this.D[l]! );
            }
            
            /** Input Batch Fixing */
            this.X[0] = batch.x;
            const Y = batch.y;
            if(epoch % 20 == 0) console.time("Inference");
            /** Inference Update Loop */
            for(let t = 1; t <= T_infer; t++){
                tf.tidy(() => {
                    for(let l = 0; l < this.L; l++){
                        /** Pre Activation Predicted Latent Values */
                        const A_l = tf.matMul(this.X[l+1], this.W[l], false, true) satisfies tf.Tensor2D;
                        /** Predicted Latent Values */
                        const Xhat_l = this.f(A_l);
                        this.keepAssign(this.E, l, tf.sub(this.X[l], Xhat_l));
                        this.keepAssign(this.H, l, tf.mul(this.E[l], this.df(A_l)))
                    }

                    let Yhat = tf.matMul(this.X[this.L], this.W[this.L+1], false, true);
            
                    let E_sup = tf.sub(Yhat, Y);
                    this.keepAssign(this.E, this.L, tf.matMul(E_sup, this.W[this.L+1]));

                    for(let l = 1; l <= this.L; l++){
                        /** Gradients of Latents */
                        const G_xl = tf.sub(this.E[l], tf.matMul(this.H[l-1], this.W[l-1]));
                        this.keepAssign(this.X, l, tf.sub(this.X[l], tf.mul(eta_infer, G_xl)));
                    }
                });
            }
            if(epoch % 20 == 0)console.timeEnd("Inference")
            if(epoch % 20 == 0)console.time("Weights")
            /** Weight Update Loop */
            for(let t = 1; t <= T_learn; t++){
                tf.tidy(() => {
                    for(let l = 0; l < this.L; l++){
                        const A_l = tf.matMul(this.X[l+1],this.W[l], false, true) satisfies tf.Tensor2D;
                        const Xhat_l = this.f(A_l);
                        this.keepAssign(this.E, l, tf.sub(this.X[l], Xhat_l));
                        this.keepAssign(this.H, l, tf.mul(this.E[l],this.df(A_l)))
                        /** Average Gradients of Weights */
                        const G_wl = tf.mul(negInvB, tf.matMul(this.H[l].transpose(), this.X[l+1]))
                        this.W[l].assign(tf.sub(this.W[l], tf.mul(eta_learn, G_wl)));
                    }
                    let Yhat = tf.matMul(this.X[this.L], this.W[this.L+1], false, true);
                    let E_sup = tf.sub(Yhat, Y);
                    /** Average Gradients of output weight */
                    const G_w_out = tf.mul(invB,tf.matMul(E_sup.transpose(), this.X[this.L]));
                    this.W[this.L+1].assign(tf.sub(this.W[this.L+1], tf.mul(eta_learn, G_w_out))); 
                });
            }
            if(epoch % 20 == 0)console.timeEnd("Weights")
            if(epoch % 20 == 0)console.log(`${epoch} / ${input.epochs}`)
            negInvB.dispose();
            invB.dispose();
        }
    }
        console.timeEnd("Training");
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
            tf.tidy(() => {
                for(let l = 0; l < this.L; l++){
                    /** Pre Activation Predicted Latent Values */
                    const A_l = tf.matMul(this.X[l+1], this.W[l], false, true) satisfies tf.Tensor2D;
                    /** Predicted Latent Values */
                    const Xhat_l = this.f(A_l);
                    this.keepAssign(this.E, l, tf.sub(this.X[l], Xhat_l));
                    this.keepAssign(this.H, l, tf.mul(this.E[l], this.df(A_l)))
                }            
                this.keepAssign(this.E, this.L, zeros(xBatch.shape[0]!,this.D[this.L]!))

                for(let l = 1; l <= this.L; l++){
                    /** Gradients of Latents */
                    const G_xl = tf.sub(this.E[l], tf.matMul(this.H[l-1], this.W[l-1]));
                    this.keepAssign(this.X, l, tf.sub(this.X[l], tf.mul(eta_infer, G_xl)));
                }
            });
        }
        return tf.matMul(this.X[this.L], this.W[this.L + 1], false, true) satisfies tf.Tensor2D;
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
    epochs: 200,
    T_infer: 50,
    eta_infer: 0.05,
    T_learn: 50,
    eta_learn: 0.005,
    data: [{ x, y, }],
};

let pcn = new PCN([2,6,4,2]);
pcn.Train(pcnSetup);
document.getElementById("retry")?.addEventListener('click', () => {
    const result = pcn.Generate(x, 50, 0.05);
    const resultClass = argMax(result).dataSync();
    const targetClass = argMax(y).dataSync();
    console.log("Result:", Array.from(resultClass));
    console.log("Target:", Array.from(targetClass));
});
function argMax(mat: tf.Tensor2D): tf.Tensor1D{
    return tf.argMax(mat, 1);
}

function zeros(rows: number, cols: number): tf.Tensor2D{
    return tf.zeros([rows, cols]);
}
function xavierUniform(rows: number, cols: number): tf.Tensor2D {
    const limit = Math.sqrt(6 / (rows + cols));
    return tf.randomUniform([rows, cols], -limit, limit);
}
