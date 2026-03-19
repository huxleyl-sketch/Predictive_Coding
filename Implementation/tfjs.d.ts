declare module "@tensorflow/tfjs" {
  export type Tensor2D = any;
  export function tensor2d(...args: any[]): any;
  export function matMul(...args: any[]): any;
  export function sub(...args: any[]): any;
  export function mul(...args: any[]): any;
  export function scalar(...args: any[]): any;
}
