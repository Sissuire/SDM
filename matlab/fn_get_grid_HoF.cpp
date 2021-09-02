#include "Improved_TVQA_fun.h"
#include "Improved_TVQA_def.h"
#include "mex.h"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
    // check input and output arguments.
    if( nrhs != 2 )
        mexErrMsgTxt("Error: Input Arguments.\n");  // input two frames
    if( nlhs > 1 )
        mexErrMsgTxt("Error: Output Arguments.\n");  // output desc & size
    
    // Get input data.
    double *ptr_prev = mxGetPr(prhs[0]);
    double *ptr_next = mxGetPr(prhs[1]);
    int rows = mxGetM(prhs[0]);
    int cols = mxGetN(prhs[0]);

    // run TVQA Model
    std::vector<double> desc_out;
    run(ptr_prev, ptr_next, rows, cols, desc_out);
    
    // Output.
    size_t dims = desc_out.size();
    
    plhs[0] = mxCreateDoubleMatrix(1, dims, mxREAL);
    
    double *ptr = mxGetPr(plhs[0]);
    memcpy(ptr, &desc_out[0], dims * sizeof(double));

    return;
}