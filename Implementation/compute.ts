type Matrix = Array<Array<number>>;
import { pcn } from "./main";

//L should be defined
let l = 0;

export const wgsl = /* wgsl */`

/*

for(let t = 1; t <= this.T_infer; t++){
    let H: Matrix[] = [];
    //prediction
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
*/

@group(0) @binding(0) var<storage, read_write> /* Edges */ E : array<f32, /* Layers */ ${pcn.L}>

@compute @workgroup_size(/* width */ ${pcn.B}, /* height */ ${pcn.D[l]}) 
fn prediction(@builtin(global_invocation_id) id : vec3u) -> array<f32,/*width * height */ ${pcn.B * pcn.D[l]}>{
    for( let j = 0; )
    let A[i + j * ${pcn.B}] = 
    /* For each x,y coordinate do */
     return array<f32,2500>
}
`