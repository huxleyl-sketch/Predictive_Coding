/*


*/

@compute @workgroup_size(/* width */ 50, /* height */ 50, /* Layers */ 500) 
fn train(@builtin(global_invocation_id) id : vec3u){
    /* For each x,y,z coordinate do */
}