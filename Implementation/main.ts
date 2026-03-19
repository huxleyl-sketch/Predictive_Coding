import * as tf from '@tensorflow/tfjs';

/**
 * Row priority Matrix  
 * [  
 * [0, 1, 2],  row 1
 * [3, 4, 5],  row 2
 * [6, 7, 8],  row 3
 * ]  
 */
type Matrix = number[][];

tf.

type PCNconfig = {
    batchSize: number,
    inputBatch: Matrix,
    targetBatch: Matrix,
    T_infer: number,
    n_infer: number,
    T_learn: number,
    n_learn: number,
    dimensions: number[],
}

class PCN {
    
    /** Gradients of Latents */
    G_x: Matrix[];
    /** Average Gradients of Weights */
    G_w: Matrix[];
    /** Inference Steps per Sample */
    T_infer: number;
    /** Inference Rate */
    n_infer: number;
    /** Learning Steps per Batch */
    T_learn: number;
    /** Learning Rate */
    n_learn: number = 0.005;
    /** Latents */
    X: Matrix[];
    /** Weights */
    W: Matrix[];
    /** Target Batch */
    Y: Matrix;
    /** Pre Activation Predicted Latent Values */
    A: Matrix[];
    /** Activation Function */
    f: (m: Matrix) => Matrix;
    /** Derivative of Activation Function */
    df: (m: Matrix) => Matrix;
    /** Predicted Latent Values */
    Xhat: Matrix[];
    /** Error */
    E: Matrix[];
    /** layers */
    L: number;
    /** Batch Size */
    B: number;
    /** Dimensions */
    D: number[]

   


    constructor(input: PCNconfig = {
        batchSize: 500,
        inputBatch: [],
        targetBatch: [],
        T_infer: 50,
        n_infer: 0.05,
        T_learn: 500,
        n_learn: 0.005,
        dimensions: [],

    }){
        this.f = (m: Matrix) => m.map( row => row.map(x => Math.max(0,x)))
        this.df = (m: Matrix) => m.map( row => row.map(x => x > 0 ? 1 : 0))

        this.A = [];
        this.Xhat = [];
        this.E = [];

        this.G_x = [];
        this.G_w = [];

        this.X = [];
        this.W = [];

        this.T_infer = input.T_infer;
        this.n_infer = input.n_infer;
        this.T_learn = input.T_learn;
        this.n_learn = input.n_learn;
        this.B = input.batchSize;
        this.L = input.dimensions.length;
        this.D = input.dimensions;

        /** Initialise Values */
        for( let l = 0; l < this.L; l++){
            this.W[l] = tf.xavierUniform(this.D[l],this.D[l + 1]);
            if(l > 0 && l < this.L ){
                /** Small Random Values */
                this.X[l] = initMatrix(this.B,this.D[l]);
            }   
        }
        this.W[this.L + 1] = initMatrix(this.D[this.L + 1], this.D[this.L])

        /** Input Batch Fixing */
        this.X[0] = input.inputBatch;
        this.Y = input.targetBatch;

    }
    Train(){
        this.InterferenceLoop();
        this.WeightsLoop();
    }

    /** Inference Update Loop */
    InterferenceLoop(){
        
        for(let t = 1; t <= this.T_infer; t++){
            let H: Matrix[] = [];

            for(let l = 0; l < this.L; l++){
                this.A[l] = mMult(this.X[l+1], Trans(this.W[l]));
                this.Xhat[l] = this.f(this.A[l]);
                this.E[l] = Sub(this.X[l], this.Xhat[l]);
                H[l] = Hadamard(this.E[l], this.df(this.A[l]))
            }

            let Yhat = mMult(this.X[this.L], Trans(this.W[this.L+1]));
            let E_sup = Sub(Yhat, this.Y);
            this.E[this.L] = mMult(E_sup, this.W[this.L+1]);

            for(let l = 1; l <= this.L; l++){
                this.G_x[l] = Sub(this.E[l], mMult(H[l-1], this.W[l-1]));
                this.X[l] = Sub(this.X[l], sMult(this.n_infer, this.G_x[l]));
            }
        }
    }
    /** Weight Update Loop */
    WeightsLoop(){
        
        for(let t = 1; t <= this.T_learn; t++){
            let H: Matrix[] = [];
            
            for(let l = 0; l < this.L; l++){
                this.A[l] = mMult(this.X[l+1],Trans(this.W[l]));
                this.Xhat[l] = this.f(this.A[l]);
                this.E[l] = Sub(this.X[l], this.Xhat[l]);
                H[l] = Hadamard(this.E[l],this.df(this.A[l]))
                this.G_w[l] = sMult(-1/this.B, mMult(Trans(H[l]), this.X[l+1]))
                this.W[l] = Sub(this.W[l], sMult(this.n_learn, this.G_w[l]));
            }

            let Yhat = mMult(this.X[this.L], Trans(this.W[this.L+1]));
            let E_sup = Sub(Yhat, this.Y);
            this.G_w[this.L + 1] = sMult(1/this.B,mMult(Trans(E_sup), this.X[this.L]));
            this.W[this.L+1] = Sub(this.W[this.L+1], sMult(this.n_learn, this.G_w[this.L + 1])); 
        }
    }
}

let pcnSetup: PCNconfig = {
    batchSize: 500,
        inputBatch: [],
        targetBatch: [],
        T_infer: 50,
        n_infer: 0.05,
        T_learn: 500,
        n_learn: 0.005,
        dimensions: [],
};
export let pcn = new PCN(pcnSetup);

function sMult(S: number, M: Matrix): Matrix {
    return M.map( row => row.map( x => S * x ) )
}
function mMult(M1: Matrix, M2: Matrix): Matrix{
    if(M1[0].length != M2.length) 
        throw new Error(`Invalid Matrix bounds M1 width: ${M1[0].length} M2 height: ${M2.length}`)
    let M2T = Trans(M2);
    return M1.map( (row) => M2T.map( (col) => row.reduce( (acc, x, i) => acc + x * col[i] , 0 ) ) )
}
function Trans(M: Matrix): Matrix {
    let Mcopy = [...M];
    return M.map( ( row, i ) => row.map( ( x, j) => Mcopy[j][i] ))
}
function Sub(M1: Matrix, M2: Matrix): Matrix {
    return M1.map( (row, i) => row.map( (x, j) => x - M2[i][j] ) )
}
function Hadamard(M1: Matrix, M2: Matrix): Matrix {
    return M1.map( (row, i) => row.map( (x, j) => x * M2[i][j] ) )
}

/** Array W x H with values [0,1] */
function initMatrix(W:number, H:number): Matrix { 
    return xavierUniform(W,H);
}

function xavierUniform(rows: number, cols: number): number[][] {
    const limit = Math.sqrt(6 / (rows + cols));
    const out: number[][] = [];

    for (let i = 0; i < rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < cols; j++) {
            row.push((Math.random() * 2 - 1) * limit);
        }
        out.push(row);
    }

    return out;
}