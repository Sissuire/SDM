#include "Improved_TVQA_fun.h"
#include "Improved_TVQA_def.h"
#include "mex.h"

void mexFunction (int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
    // check input and output arguments.
    if( nrhs != 1 )
        mexErrMsgTxt("Error: Input Arguments.\n");  // input two frames
    if( nlhs > 1 )
        mexErrMsgTxt("Error: Output Arguments.\n");  // output desc & size
    
    // Get input data.
    double *ptr_in = mxGetPr(prhs[0]);
    int rows = (int)mxGetM(prhs[0]);
    int cols = (int)mxGetN(prhs[0]);

    // run TVQA Model
    cv::Mat out;
    detect_kp(ptr_in, out, rows, cols);
    
    // Output.
    size_t dims = size_t(rows * cols);
    
    plhs[0] = mxCreateDoubleMatrix(cols, rows, mxREAL);
    
    double *ptr_out = mxGetPr(plhs[0]);
    memcpy(ptr_out, (double*)out.data, dims * sizeof(double));

    return;
}