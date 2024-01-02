#include <cupy/complex.cuh>

/*
We design a kernel to multiply arbitrary slices from an input 1-D array with
indexed arrays from a second input 2-D matrix.

Long input array x is sliced arbitrarily at different indices:
A-B, C-D, E-F, ... where each slice is no longer than length N.

Second input array y is a 2D matrix with individual rows presenting unique
inputs.

The goal is to multiply each slice with a particular row of the matrix. This is specified
with additional index arrays; in general this can be accomplished by iterating over
each slice and multiplying by the appropriate row, at the appropriate length.

However, this kernel attempts to exploit optimistic cases where
number of rows of y <<< number of slices from x.
This means that there is a high probability for the row to be reused from slice to slice,
preventing additional global memory reads.

Each block will allocate shared memory to hold one row, and will swap it out if there is a change
in the row index for the current slice.
*/
extern "C" __global__
void multiplySlicesWithIndexedRowsOptimistic(
    const complex<float> *d_x,
    const int xlength,
    const complex<float> *d_rows,
    const int rowLength,
    const int numRows,
    const int *d_sliceStarts, // length numSlices
    const int *d_sliceLengths, // length numSlices
    const int numSlices,
    const int *d_rowIdxs, // length numSlices
    complex<float> *d_out, // numSlices * rowLength
    int outlength)
{
    // allocate shared memory
    extern __shared__ double s[];

    complex<float> *s_row = (complex<float>*)s; // (rowLength) complex floats

    // On initialization, the loaded index is set to an invalid value
    int loadedRow = -1;
    int requiredRow;

    // Allocate stack variables to hold the current slice indices
    int sliceStart, sliceLength;

    // Iterate over the slices
    for (int i = blockIdx.x; i < numSlices; i += gridDim.x) // each block computes 1 slice at a time
    {
        // First we read the required row index
        requiredRow = d_rowIdxs[i];

        // Then we update shared memory if it's required
        if (requiredRow != loadedRow)
        {
            for (int t = threadIdx.x; t < rowLength; t = t + blockDim.x)
            {
                s_row[t] = d_rows[requiredRow * rowLength + t];
            }

            // Wait for it to be fully loaded
            __syncthreads();
        }

        // Then we perform the multiplies
        sliceStart = d_sliceStarts[i];
        sliceLength = d_sliceLengths[i];

        // Write the output to global mem
        for (int t = threadIdx.x; t < sliceLength; t += blockDim.x)
        {
            if (sliceStart + t >= 0 && sliceStart + t < xlength)
                d_out[i*outlength + t] = s_row[t] * d_x[sliceStart + t];    
        }
            
        
    }

 
}


/*
Here we design a sliding template multiply, with an additional sliding norm calculation.
This is usually used in the xcorr step.

We have two inputs:
1) x: A short (can be contained in shared mem) template array, length xlen
2) y: Another arbitrarily long input array to slide against

This is the MOST OPTIMISTIC method for the copies;
we assume that within the shared memory we can fit both
1) x itself
2) a large section of y

This allows us to slide against multiple xlen windows within 1 block,
escaping a lot of repeated global memory reads.

This is especially important when the template itself is very short;
as the template gets longer this matters less and less when compared to the next 
step in the xcorr, which is the FFT step.

Optimisation rule of thumb:
Use enough numSlidesPerBlk to spawn a decent number of blocks to fill SMs; 
too many SlidesPerBlk will result in too little SM usage.
Threads can generally be left at around 128 (or less if x is smaller than 128 elements)

*/
extern "C" __global__ 
void slidingMultiplyNormalised(
    const complex<float> *x, // the template
    const int xlen,
    const complex<float> *y, // the searched array
    const int ylen,
    const int startIdx, // the start index of the searched array to begin the sliding
    const int idxlen, // the total number of slides
    complex<float> *z, // the output array, which has dimensions (idxlen) rows * (xlen) columns, and is normalised by the normsq of the corresponding slice of y
    int numSlidesPerBlk, // this defines the number of slides to compute per block, and hence determines the workspace (which the caller must calculate correctly)
    const double* coefficient // extra coefficient to multiply by (usually this is just the pre-computed energy i.e. normSq of x, the template)
){
    // allocate shared memory
    extern __shared__ double s[];

    complex<float> *s_x = (complex<float>*)s; // (xlen) complex floats
    complex<float> *s_ysection = (complex<float>*)&s_x[xlen]; // (numSlidesPerBlk + xlen - 1) complex floats
    double *s_ws = (double*)&s_ysection[numSlidesPerBlk + xlen - 1]; // (blockDim.x) doubles

    // Load shared mem x and y
    for (int t = threadIdx.x; t < xlen; t += blockDim.x)
        s_x[t] = x[t];

    // Initialize workspace to 0s (workspace is assumed equal to blocksize)
    s_ws[threadIdx.x] = 0.0;

    int ysectionSize = numSlidesPerBlk + xlen - 1;
    int ysectionOffset = blockIdx.x * numSlidesPerBlk;
    // The number of slides computed this block; the last block may compute less than the allocation
    int numSlidesThisBlk = idxlen - ysectionOffset > numSlidesPerBlk ? numSlidesPerBlk : idxlen - ysectionOffset;

    complex<float> yt;
    for (int t = threadIdx.x; t < ysectionSize; t += blockDim.x)
    {
        if (ysectionOffset + t < ylen) // read within bounds
        {
            yt = y[ysectionOffset + t + startIdx];
            s_ysection[t] = yt;
            // Initialize the first normSq workspace values while reading in
            if (t < xlen) // Each thread accumulates in its own index in the workspace
                s_ws[threadIdx.x] += (double)norm(yt); // 'norm' is actually magn squared -> See cupy/cupy/_core/include/cupy/complex/complex.h
            // Accumulate as doubles to maintain some good precision
        }
        else
            s_ysection[t] = complex<float>(0.0f, 0.0f);
    }
     
    // Wait for shared mem syncs
    __syncthreads();

    // Perform first normSq calculation via reduction in the workspace
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            s_ws[threadIdx.x] += s_ws[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Define stack-variables for the thread
    int row_offset;
    complex<float> zt;
    float normalisation;

    /*
    Compute loop.
    It has been benchmarked that the reverse loop is slower i.e.
    outer loop threadIdx, inner loop slidesPerBlk.

    This is probably due to increased impact of repeated norm calls; when the block goes to
    the next iteration, it has to start from the first normSq again and re-do the same normSq calculations.
    This trumps any reduced shared mem reads we might benefit from.
    */

    // Begin the sliding multiplies; outer loop 
    // Now extract the first normSq value for the first slide in this block
    double normSq = s_ws[0]; // broadcast to everyone
    for (int i = 0; i < numSlidesThisBlk; i++) // we only compute the necessary number of slides
    {
        // Define the starting output index for this slide (skip the block, then skip the index as well)
        row_offset = blockIdx.x * numSlidesPerBlk + i;

        // Re-calculate the norm for this slide
        if (i != 0)
        {
            // Remove the energy of the element before the start,
            // and add the energy of the element at the end
            normSq = normSq - (double)norm(s_ysection[i-1]) + (double)norm(s_ysection[i+xlen-1]);
        }
        // Compute normalisation with the coefficient
        normalisation = (float)(sqrt(normSq) * *coefficient); // note that we later divide by the norm, not the normSq!

        // Inner loop: Simply multiply and write to global mem, doing it this way lets us conveniently write to contiguous global mem
        for (int t = threadIdx.x; t < xlen; t += blockDim.x)
        {
            zt = s_x[t] * s_ysection[i+t] / normalisation; // this is the bulk of the compute time
            z[row_offset * xlen + t] = zt; // This should be global coalesced? 
            // Note that when timing using nsys, the compiler may have optimized away the computation line if
            // the global write has been commented out. This may make it appear like the global write is the one taking a long time.
        }
            
    } // End outer loop
}

/*
Multi-template sliding dot products.

Motivation:
To discard the use of cuFFT calls, one can create multiple templates
to test for with correlation; these will come with their frequency shifted copies.
This way you will not have to use FFT to search the CAF.

However, this assumes some knowledge of the frequency range which the peak will reside in.

Also, this can be used to test multiple distinct templates.

Only the best template (index) should be saved, along with the normalised peak value.

Example:
2 templates with 3 frequency shifts -> 6 total templates.

Design goals:
Use shared memory to store all templates.

Recommendation:
This may not perform well with large OSRs, especially since it will consume a lot of shared memory.

Algorithm:
1) Load a slice of the input array.
2) Loop over following:
3) Load a single template into shared mem.
4) Slide and dot product template across the input slice.
5) For each one, save dot product into shared memory workspace if higher than before, and also write the current template index.
6) Repeat 2-5 for each template.
7) Write shared mem workspace to final global output.
*/

extern "C" __global__ 
void multiTemplateSlidingDotProduct(
    const complex<float> *d_templates, // (numTemplate rows * templateLength columns)
    const float *d_template_energies,
    const int numTemplates,
    const int templateLength,
    const complex<float> *d_x,
    const int xlength,
    const int numSlidesPerBlk,
    int *d_templateIdx,
    float *d_output
){
    const int xsectionLength = numSlidesPerBlk + templateLength - 1;

    // allocate shared memory
    extern __shared__ double s[];

    complex<float> *s_template = (complex<float>*)s; // (templateLength) complex floats
    complex<float> *s_xsection = (complex<float>*)&s_template[templateLength]; // (numSlidesPerBlk + templateLength - 1) complex floats
    complex<float> *s_ws = (complex<float>*)&s_xsection[xsectionLength]; // (THREADS_PER_BLK) complex floats
    int *s_templateIdx = (int*)&s_ws[blockDim.x]; // (numSlidesPerBlk) ints
    float *s_output = (float*)&s_templateIdx[numSlidesPerBlk]; // (numSlidesPerBlk) floats
    /*
    Total shared memory footprint:
    numSlidesPerBlk * (4 + 4 + 8) + 
    templateLength * (8 + 8) + 
    THREADS_PER_BLK * 8

    Ignored the -1 from xsection
    */

    float *s_wsf = (float*)s_ws; // we mangle the pointer type here just for the pre-normalization calculation

    // Initialize the xsection slice
    for (int t = threadIdx.x; t < xsectionLength; t += blockDim.x)
    {
        int idx = blockIdx.x * numSlidesPerBlk + t; // the accessor index for the global input
        complex<float> xsection = (idx < xlength) ? d_x[idx] : 0; // read to register
        s_xsection[t] = xsection; // write to shared mem

        // We also fill up the workspace to calculate the first normalization here
        if (t < templateLength)
        {
            // we use our mangled pointer here..
            if (t == threadIdx.x) // first loop we write
                s_wsf[threadIdx.x] = norm(xsection); // 'norm' is actually magn squared -> See cupy/cupy/_core/include/cupy/complex/complex.h
            else // consequent loops we accumulate
                s_wsf[threadIdx.x] += norm(xsection);
        }
            
    }

    // Initialize the shared memory temporary outputs
    for (int t = threadIdx.x; t < numSlidesPerBlk; t += blockDim.x)
    {
        s_templateIdx[t] = 0;
        s_output[t] = 0.0f;
    }

    __syncthreads();

    // Pre-Compute the first normalization across the xsection slice
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            // accumulate using the mangled pointer
            s_wsf[threadIdx.x] += s_wsf[threadIdx.x + s];
        }
        __syncthreads();
    }

    // All threads keep a copy of the first normalization in register
    float firstNormalization = s_wsf[0];
    float normalization;

    //////// Begin computation loop
    for (int i = 0; i < numTemplates; i++)
    {
        // Load the current template into the shared memory
        for (int t = threadIdx.x; t < templateLength; t += blockDim.x)
            s_template[t] = d_templates[i * templateLength + t];

        // And the current energy (into thread local register)
        float template_energy = d_template_energies[i];
        
        __syncthreads();

        // Iterate over the slides
        for (int k = 0; k < numSlidesPerBlk; k++)
        {
            // Multiply across the template
            for (int t = threadIdx.x; t < templateLength; t += blockDim.x)
            {
                // On first loop write
                if (t == threadIdx.x)
                    s_ws[threadIdx.x] = s_template[t] * s_xsection[k + t];
                // On consequent loops accumulate
                else
                    s_ws[threadIdx.x] += s_template[t] * s_xsection[k + t];
            }
            
            // At this point the workspace needs to be summed over
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (threadIdx.x < s)
                {
                    s_ws[threadIdx.x] += s_ws[threadIdx.x + s];
                }
                __syncthreads();
            }
            // Now the dot product is in the first index

            // Update the shared memory output if better
            if (threadIdx.x == 0)
            {
                // Update the normalization energy for the current slide
                // Each slide works on indices[k, k+templateLength)
                normalization = (k > 0) ? normalization + norm(s_xsection[k + templateLength-1]) - norm(s_xsection[k-1]) : firstNormalization;

                // Calculate the output with both template and input normalizations
                float output = norm(s_ws[0]) / template_energy / normalization; // 'norm' is actually magn squared -> See cupy/cupy/_core/include/cupy/complex/complex.h

                // Only overwrite if larger
                if (output > s_output[k])
                {
                    s_output[k] = output;
                    s_templateIdx[k] = i;
                }
            }
        }
    }

    // When complete, write to global output
    __syncthreads(); // do i need this one?
    for (int t = threadIdx.x; t < numSlidesPerBlk; t += blockDim.x)
    {
        d_output[blockIdx.x * numSlidesPerBlk + t] = s_output[t];
        d_templateIdx[blockIdx.x * numSlidesPerBlk + t] = s_templateIdx[t];
    }

}