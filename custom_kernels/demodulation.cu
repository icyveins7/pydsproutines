#include <cupy/complex.cuh>

/*
This kernel is designed to work on one 'signal' for each block.
A signal is defined by a row from a matrix, with each element representing a symbol.
E.g. for QPSK the values would be {0,1,2,3}, defined as unsigned chars i.e uint8.
Each signal takes a reference from an index matrix (see argmax.cu),
 where each row corresponds to the following in order:
A: key index
B: sample index
C: rotation

The key index references an array of key lengths, where the slice from the input signal is made as follows:
x[keyLengths[A] + B : N]
where N is a desired stop sample, extracted from a separate array,
 specific to each block i.e. each signal.

The output is gray-mapped in the following configuration:

    1   |   0            1 (01)  |  3 (11)    
    ---------   ====>    -----------------
    2   |   3            0 (00)  |  2 (10)

Original scenario:
    Given a number of preambles, test each signal against all preambles (compareIntPreambles kernel)
    and then find the best preamble match(A), the preamble starting index (B), and the
    rotation required to fit the preamble (C); see argmax.cu.

    This kernel then extracts the signal (minus the preamble) up to a certain desired length, for each signal,
    performing the rotation C in the process. After this, the resulting symbols are in the correct
    rotation order, and have been cut to the exact length stipulated, so they are ready to be decoded/interpreted.

*/
extern "C" __global__
void cutAndRotatePSKSymbolsFromPossiblePreambles_Gray(
    const unsigned int *d_indexMatrix, // numRows x 3
    const int numRows,
    const unsigned char *d_syms, // numRows * symsLength
    const int symsLength,
    const unsigned int *d_keyLengths,
    const unsigned int *d_sampleStops, // numRows
    const unsigned char m, // modulation order e.g. BPSK = 2, QPSK = 4, 8PSK = 8
    const int outLength,
    unsigned char * d_out, // numRows * outlength
    unsigned int *d_count // numRows or NULL
){
    // Exit if block index is more than the row number
    if (blockIdx.x >= numRows)
        return;

    // Allocate shared memory to read in the index matrix row
    extern __shared__ double s[];

    unsigned int *s_indexRow = (unsigned int*)s; // (3) unsigned ints

    if (threadIdx.x < 3)
        s_indexRow[threadIdx.x] = d_indexMatrix[blockIdx.x * 3 + threadIdx.x];
    __syncthreads();

    // Interpret the 3 values
    const unsigned int keyLength = d_keyLengths[s_indexRow[0]];
    const unsigned int sampleStart = s_indexRow[1] ;
    const unsigned char rotation = (const unsigned char)s_indexRow[2];
    
    // Read the length requirement for this row
    const unsigned int sampleStop = d_sampleStops[blockIdx.x];

    // Get the pointer to the input row for this block
    const unsigned char *syms = &d_syms[symsLength * blockIdx.x];
    // And the output row for this block
    unsigned char *out = &d_out[outLength * blockIdx.x];
    // Compute the offset for this block
    const unsigned int offset = keyLength + sampleStart;
    unsigned int totalWrite = sampleStop - offset;
    // Extract, rotate, and write
    unsigned char val;
    // Gray mapping
    unsigned char gray[4] = {3,1,0,2};

    for (int t = threadIdx.x; t < totalWrite; t += blockDim.x)
    {
        // Make sure in bounds for both read and write
        if (t < outLength && offset + t < symsLength)
        {
            val = syms[offset + t]; // Read
            val = (val + rotation) % m; // Rotate
            val = gray[val]; // Gray map
            out[t] = val; // Write
        }
    }

    if (threadIdx.x == 0 && d_count != NULL)
        d_count[blockIdx.x] = totalWrite;
}

/*
Assumes shared memory is sufficiently allocated;
The 2x2 result lies at the starting 4 elements.
The 2 eigenvalues lie right after. Bigger one is first.
The 2 eigenvectors are at the end.

As this is a __device__ function, you should use this as a building block
in a bigger kernel, bearing in mind the shared mem requirements.
*/
extern "C" __device__
void eigDecomposeInSharedMem_2x2(
    const complex<float> *s_x, // Shared memory, xlength * 8
    float *s_ws, // blockDim x 4 floats, must be pre-zeroed
    const int xlength,
    const int factor // number of times to square the signal before eigendecomposition
){
    // loop over the signal
    complex<float> x;
    for (int i = threadIdx.x; i < xlength; i += blockDim.x)
    {
        // Extract from sharedmem
        x = s_x[i];

        // perform squaring as necessary
        for (int f = 0; f < factor; f++)
            x = x * x;

        // Now calculate the different components
        s_ws[threadIdx.x * 4 + 0] += x.real() * x.real();
        s_ws[threadIdx.x * 4 + 1] += x.real() * x.imag();
        s_ws[threadIdx.x * 4 + 2] += x.real() * x.imag();
        s_ws[threadIdx.x * 4 + 3] += x.imag() * x.imag();
    }   
    __syncthreads();
    
    // gather into 2x2 at the front
    if (threadIdx.x < 4)
    {
        // remember that we can skip the first 2x2 values
        for (int i = threadIdx.x + 4; i < blockDim.x * 4; i += 4)
        {
            s_ws[threadIdx.x] += s_ws[i];
        }
    }
    __syncthreads(); // we cannot place syncthreads in a conditional block!
        
    // hence split the conditional into this next section again
    if (threadIdx.x < 4)
    {
        // perform the 2x2 eigen decomposition
        float T = s_ws[0] + s_ws[3]; // trace
        float D = s_ws[0] * s_ws[3] - s_ws[1] * s_ws[2]; // determinant
        
        float p1 = T/2.0;
        float p2 = sqrtf(fmaf(p1, p1, -D));
        
        // 0 and 2 write the first eigenvector
        if (threadIdx.x % 2 == 0)
        {
            // compute the eigenvalue
            float l1 = p1 + p2;
            
            // 0 writes the eigenvalue
            if (threadIdx.x == 0){s_ws[4] = l1;}
            
            // compute the eigenvector
            s_ws[6+threadIdx.x] = (threadIdx.x == 0) ? (l1 - s_ws[3]) : s_ws[2];
        }
        else // 1 and 3 write the second eigenvector
        {
            // compute the eigenvalue
            float l2 = p1 - p2;
            
            // 1 writes the eigenvalue
            if (threadIdx.x == 1){s_ws[5] = l2;}
            
            // compute the eigenvector
            s_ws[6+threadIdx.x] = (threadIdx.x == 1) ? (l2 - s_ws[3]) : s_ws[2];
        }
        
    }
    __syncthreads();
}

// This is a sample kernel to check the outputs of the above kernel.
// Not meant for actual use.
extern "C" __global__
void detectBPSKorQPSK(
    const complex<float> *d_x,
    const int xlength,
    const int numSignals,
    float *d_eigresults // each block writes 10 values
){
    // Allocate shared memory to read in the signal (one row)
    extern __shared__ double s[];

    complex<float> *s_x = (complex<float>*)s; // (xlength) complex floats
    float *s_ws = (float*)&s_x[xlength]; // (blockDim.x * 4) floats

    // Read row for the block
    for (int t = threadIdx.x; t < xlength; t += blockDim.x)
        s_x[t] = d_x[blockIdx.x * xlength + t];

    // Pre-zero workspace
    for (int t = threadIdx.x; t < blockDim.x * 4; t += blockDim.x)
        s_ws[t] = 0.0f;

    __syncthreads(); // wait for shared mem to be loaded

    // Call the device func
    eigDecomposeInSharedMem_2x2(
        s_x, // Shared memory, xlength * 8
        s_ws, // blockDim x 4 floats
        xlength,
        0 // number of times to square the signal before eigendecomposition
    );

    // Write results
    if (threadIdx.x < 10)
    {
        d_eigresults[blockIdx.x * 10 + threadIdx.x] = s_ws[threadIdx.x];
    }
}


extern "C" __device__
void demod_bpsk_in_shared(
    complex<float> *s_x, // xlength
    complex<float> *s_ws, // blockDim.x
    const int xlength,
    unsigned char *d_syms_exact // the exact starting pointer; if multiple rows output, point this to the exact row
){
    // Take the power 2 of each sample, and sum over them for each thread
    complex<float> t_xtotal = 0;
    complex<float> t_x;
    for (int t = threadIdx.x; t < xlength; t += blockDim.x)
    {
        t_x = s_x[t] * s_x[t]; // square it

        // sum for the current thread
        t_xtotal += t_x;
    }
    // At the end, we write the thread's total to its spot in the shared mem workspace
    s_ws[threadIdx.x] = t_xtotal;

    // Wait for all threads to finish writing to workspace
    __syncthreads();

    // Sum over the workspace & compute the angle correction
    t_xtotal = 0;
    for (int i = 0; i < blockDim.x; i++){
        t_xtotal += s_ws[i]; // all threads read the same memory address from shared mem workspace
    }
    float angleCorrection = atan2f(t_xtotal.imag(), t_xtotal.real()) / 2; // we divide by 2 to counter the power of 2 we induced

    // Generate the phase correction
    float real, imag;
    sincosf(-angleCorrection, &imag, &real); // we correct to 0
    complex<float> e(real, imag);

    // Apply the phase correction and map the symbol
    int xsign;

    for (int i = threadIdx.x; i < xlength; i += blockDim.x)
    {
        // Each thread reads its own value and stores in its own register (do not write back to shared mem! no point!)
        t_x = s_x[i] * e;

        // Map the symbol based on sign of real/imag
        xsign = signbit(t_x.real()); // signbit returns positive->0, negative->1, which corresponds to the actual sign bit value

        // We simply want to return the sign bit, since it fits our mapping

        // Write to global mem
        d_syms_exact[i] = (unsigned char)xsign;
    }
}

extern "C" __device__
void demod_qpsk_in_shared(
    complex<float> *s_x, // xlength
    complex<float> *s_ws, // blockDim.x
    const int xlength,
    unsigned char *d_syms_exact // the exact starting pointer; if multiple rows output, point this to the exact row
){
    // Take the power 4 of each sample, and sum over them for each thread
    complex<float> t_xtotal = 0;
    complex<float> t_x;
    for (int t = threadIdx.x; t < xlength; t += blockDim.x)
    {
        t_x = s_x[t] * s_x[t]; // square it
        t_x = t_x * t_x; // square it again to get power 4
        // sum for the current thread
        t_xtotal += t_x;
    }
    // At the end, we write the thread's total to its spot in the shared mem workspace
    s_ws[threadIdx.x] = t_xtotal;

    // Wait for all threads to finish writing to workspace
    __syncthreads();

    // Sum over the workspace & compute the angle correction
    t_xtotal = 0;
    for (int i = 0; i < blockDim.x; i++){
        t_xtotal += s_ws[i]; // all threads read the same memory address from shared mem workspace
    }
    float angleCorrection = atan2f(t_xtotal.imag(), t_xtotal.real()) / 4; // we divide by 4 to counter the power of 4 we induced

    // Generate the phase correction
    float real, imag;
    sincosf(-angleCorrection + 0.78539816340, &imag, &real); // we shift it to pi/4 for gray coding later
    complex<float> e(real, imag);

    // Apply the phase correction and map the symbol
    int xsign, ysign;
    const unsigned char mapping[2][2] = {
        {0, 3},
        {1, 2}
    };
    for (int i = threadIdx.x; i < xlength; i += blockDim.x)
    {
        // Each thread reads its own value and stores in its own register (do not write back to shared mem! no point!)
        t_x = s_x[i] * e;

        // Map the symbol based on sign of real/imag
        xsign = signbit(t_x.real()); // signbit returns positive->0, negative->1, which corresponds to the actual sign bit value
        ysign = signbit(t_x.imag()); // we also cast it so we can manipulate at the byte level later

        // We want to return a single number of {0,1,2,3} based on the following scheme (rotating counter clockwise):
        // (+, +) = (0, 0) -> 0
        // (-, +) = (1, 0) -> 1
        // (-, -) = (1, 1) -> 2
        // (+, -) = (0, 1) -> 3
        // See the const mapping array above to verify this is correct.

        // Write to global mem
        d_syms_exact[i] = mapping[xsign][ysign];
    }
}

/*
This kernel takes in a matrix of signals, with 1 signal in each row.
A block is assigned to each signal, which performs the phase-locking to the QPSK constellation,
maps the values to {0,1,2,3} based on an anticlockwise direction and then writes the output back to global memory.

NOTE: THIS MAPPING IS NOT THE GRAY CODING CONSTELLATION.

Note: the memory block for the matrix inevitably has a fixed number of columns,
but each signal may occupy less than the maximum number of columns.
The remaining columns for the signal should be zero-ed out.
*/
extern "C" __global__
void demod_qpsk(
    const complex<float> *d_x, // Input matrix, dimensions of numSignals * xlength
    const int xlength, // Number of columns in d_x (each signal may occupy less than this value, rest of the columns must be zero-ed out)
    const int numSignals, // Number of signals i.e. rows in d_x
    unsigned char *d_syms // Output matrix, dimensions also numSignals * xlength
){
    // Exit if the block number is more than the number of signals
    if (blockIdx.x >= numSignals)
        return;

    // Allocate shared memory to read in the signal (one row)
    extern __shared__ double s[];

    complex<float> *s_x = (complex<float>*)s; // (xlength) complex floats
    complex<float> *s_ws = (complex<float>*)&s_x[xlength]; // (blockDim.x) complex floats

    // Read the row assigned to the current block
    int blkGlobalOffset = blockIdx.x * xlength;
    for (int t = threadIdx.x; t < xlength; t += blockDim.x)
        s_x[t] = d_x[blkGlobalOffset + t];

    // Wait for signal to finish copying to shared memory
    __syncthreads();

    // Call the device function
    demod_qpsk_in_shared(s_x, s_ws, xlength, &d_syms[blkGlobalOffset]);
}

/*
This is a more generalised form of the demodulation function,
that takes in a secondary array that specifies the PSK order m of each row of the input.
Currently, this only evaluates m=2 (BPSK) or m=4 (QPSK), and will call the appropriate
demodulator device function.
*/
extern "C" __global__
void demod_b_or_q_psk(
    const complex<float> *d_x, // Input matrix, dimensions of numSignals * xlength
    const int xlength, // Number of columns in d_x (each signal may occupy less than this value, rest of the columns must be zero-ed out)
    const int numSignals, // Number of signals i.e. rows in d_x
    const unsigned char *d_m, // PSK order matrix, m = 2 or 4, length numSignals
    unsigned char *d_syms // Output matrix, dimensions also numSignals * xlength
){
    // Exit if the block number is more than the number of signals
    if (blockIdx.x >= numSignals)
        return;

    // Allocate shared memory to read in the signal (one row)
    extern __shared__ double s[];

    complex<float> *s_x = (complex<float>*)s; // (xlength) complex floats
    complex<float> *s_ws = (complex<float>*)&s_x[xlength]; // (blockDim.x) complex floats

    // Read the row assigned to the current block
    int blkGlobalOffset = blockIdx.x * xlength;
    for (int t = threadIdx.x; t < xlength; t += blockDim.x)
        s_x[t] = d_x[blkGlobalOffset + t];

    // Wait for signal to finish copying to shared memory
    __syncthreads();

    // Read the m value for this row
    unsigned char m = d_m[blockIdx.x];

    // Call the appropriate device function
    if (m == 2)
        demod_bpsk_in_shared(s_x, s_ws, xlength, &d_syms[blkGlobalOffset]);
    else if (m == 4)
        demod_qpsk_in_shared(s_x, s_ws, xlength, &d_syms[blkGlobalOffset]);
}


/*
This is a helper function to load the shared memory workspace with the
current block's symbols and all the preambles, used in compareIntegerPreambles
kernels and its flavours.

Seems to have errors?
TODO: either make this work or rework without it. Shared memory can only be directly invoked in kernel,
so that may be the issue.
*/
extern "C" __device__
void loadSymsAndPreamblesIntoShared(
    const double *s,
    const unsigned char *d_preambles,
    const int numPreambles,
    const int *preambleLengths,
    const unsigned char *d_syms,
    const int symsLength,
    const int m,
    const int searchStart,
    const int searchEnd,
    unsigned int *s_ws, // from this onwards are the outputs
    unsigned char *s_syms,
    unsigned char *s_preambles,
    int *s_preambleLengths,
    int *matchesSzPerPreamble,
    int *outputBlkGlobalOffset
){
    // Allocate shared memory to read in the signal (one row) and preambles
    s_preambleLengths = (int*)s; // (numPreambles) ints

    // Read in the preamble lengths first
    for (int t = threadIdx.x; t < numPreambles; t += blockDim.x){
        s_preambleLengths[t] = preambleLengths[t];
    }
    // Wait for the lengths to be read in
    __syncthreads();

    // Now check the total length required for all preambles
    int totalPreambleLength = 0;
    for (int i = 0; i < numPreambles; i++)
        totalPreambleLength += s_preambleLengths[i];

    // And then we assign the space for the rest of shared memory
    *matchesSzPerPreamble = m * (searchEnd - searchStart);
    const int matchesSzPerSignal = numPreambles * (*matchesSzPerPreamble);
    *outputBlkGlobalOffset = blockIdx.x * matchesSzPerSignal;

    s_ws = (unsigned int*)&s_preambleLengths[numPreambles]; // (m * (searchEnd-searchStart)) unsigned ints, this is the 'matches' matrix
    s_syms = (unsigned char*)&s_ws[*matchesSzPerPreamble]; // (symsLength) unsigned chars
    s_preambles = (unsigned char*)&s_syms[symsLength]; // (totalPreambleLength) unsigned chars
    /* IMPORTANT:
    We stack the shared memory in order of int32->uint32->uint8->uint8
    because CUDA enforces all pointers of a type to be aligned to an address of that type's size.
    If we moved the uint32 pointer to the last part of the shared memory,
    the uint8s may lead to a non-32-bit aligned address as the start of the next memory section,
    and this will cause a misaligned address error during execution 
    (
        NOTE: compilation will be successful,
        and the kernel will actually run without error if you get lucky!)
    */


    // Copy these into shared memory
    int blkGlobalOffset = blockIdx.x * symsLength;
    for (int t = threadIdx.x; t < symsLength; t += blockDim.x)
        s_syms[t] = d_syms[blkGlobalOffset + t]; // Copy only this row

    for (int t = threadIdx.x; t < totalPreambleLength; t += blockDim.x)
        s_preambles[t] = d_preambles[t]; // Copy the entire concatenated array of preambles
    
    // Wait for all shared mem copies to complete
    __syncthreads();
}

/*
This kernel is used to process a matrix of symbols which have been demodulated into 
unsigned char arrays of values [0, m-1] e.g. for QPSK, {0,1,2,3}.
Every block works on one signal (one row), and compares that signal to a list of preambles,
over several search indices, attempting all rotations e.g. QPSK has 4 rotations: 0->1->2->3->0.
It will return, for this signal, the best number of matches for each preamble,
its associated search index, and the rotation induced.

Notably, the preambles may be of different lengths; 
however, there must be sufficient shared memory
to hold both the signal (within the search range) and all the preambles.
As an input, the variable length preambles should be concatenated into a single long array and passed in as such.
A separate array (preambleLengths) will tell the kernel where along this long array each individual preamble occupies.
Example:
d_preambles = (preamble 1..) (preamble 2 ...............) (preamble 3........)
preambleLengths = {32, 128, 64};

Within a block, each thread will tackle one of the search indices. Hence, for searching 128 possible indices,
128 threads would be ideal. All reads at this point would be within shared memory.

The thread must scan - for its current search index - the matching rotation for each preamble sample, and increment the value.
The resulting matrix is of size (searchSize * m) where each thread increments the 'm' counters in its own row.
This is then output to global memory appropriately.

Note that the size of the output 'matches' is determined by:
(numSignals * numPreambles * m). As such, be aware that this may be excessively large
if (numSignals) is large.

An optional mask may be specified (opt_m_mask) which has a value of m for each row of d_syms
i.e. each block reads 1 value. If the input 'm' value is not equal to the block's opt_m_mask value,
the block returns without doing any work. This should allow you to only work on certain rows with certain
preambles.
*/
extern "C" __global__
void compareIntegerPreambles(
    const unsigned char *d_syms,
    const int numSignals,
    const int symsLength, 
    const int searchStart,
    const int searchEnd,
    const unsigned char *d_preambles,
    const int numPreambles,
    const int *preambleLengths,
    const int m,
    unsigned int *matches,
    unsigned char *opt_m_mask
){
    // Exit if the block number is more than the number of signals
    if (blockIdx.x >= numSignals)
        return;

    // If mask is specified, exit if the current block's mask value doesn't match
    if (opt_m_mask != NULL && (int)opt_m_mask[blockIdx.x] != m)
        return;

    // Allocate shared memory to read in the signal (one row) and preambles
    extern __shared__ double s[];
    int *s_preambleLengths = (int*)s; // (numPreambles) ints

    // Read in the preamble lengths first
    for (int t = threadIdx.x; t < numPreambles; t += blockDim.x){
        s_preambleLengths[t] = preambleLengths[t];
    }
    // Wait for the lengths to be read in
    __syncthreads();

    // Now check the total length required for all preambles
    int totalPreambleLength = 0;
    for (int i = 0; i < numPreambles; i++)
        totalPreambleLength += s_preambleLengths[i];

    // And then we assign the space for the rest of shared memory
    const int matchesSzPerPreamble = m * (searchEnd - searchStart);
    const int matchesSzPerSignal = numPreambles * matchesSzPerPreamble;
    const int outputBlkGlobalOffset = blockIdx.x * matchesSzPerSignal;

    unsigned int *s_ws = (unsigned int*)&s_preambleLengths[numPreambles]; // (m * (searchEnd-searchStart)) unsigned ints, this is the 'matches' matrix
    unsigned char *s_syms = (unsigned char*)&s_ws[matchesSzPerPreamble]; // (symsLength) unsigned chars
    unsigned char *s_preambles = (unsigned char*)&s_syms[symsLength]; // (totalPreambleLength) unsigned chars
    /* IMPORTANT:
    We stack the shared memory in order of int32->uint32->uint8->uint8
    because CUDA enforces all pointers of a type to be aligned to an address of that type's size.
    If we moved the uint32 pointer to the last part of the shared memory,
    the uint8s may lead to a non-32-bit aligned address as the start of the next memory section,
    and this will cause a misaligned address error during execution 
    (
        NOTE: compilation will be successful,
        and the kernel will actually run without error if you get lucky!)
    */


    // Copy these into shared memory
    int blkGlobalOffset = blockIdx.x * symsLength;
    for (int t = threadIdx.x; t < symsLength; t += blockDim.x)
        s_syms[t] = d_syms[blkGlobalOffset + t]; // Copy only this row

    for (int t = threadIdx.x; t < totalPreambleLength; t += blockDim.x)
        s_preambles[t] = d_preambles[t]; // Copy the entire concatenated array of preambles
    
    // Wait for all shared mem copies to complete
    __syncthreads();

    // Now we process each preamble individually
    unsigned char *s_preamble_test = s_preambles;
    
    unsigned char counterIdx;
    for (int i = 0; i < numPreambles; i++)
    {
        if (i > 0) // Increment pointer to the next preamble
            s_preamble_test = s_preamble_test + s_preambleLengths[i-1];

        // Each thread tackles a search index, looping until all search indices are complete
        for (int t = threadIdx.x; t < searchEnd - searchStart; t += blockDim.x)
        {
            int searchIdx = searchStart + t;
            
            // Pre-zero our workspace
            for (int j = 0; j < m; j++)
            {
                s_ws[t*m + j] = 0;
            }

            // Loop over the current preamble
            for (int j = 0; j < s_preambleLengths[i]; j++)
            {
                // Calculate the proper rotation by adding m to the input signal
                // Explicit upcasting
                counterIdx = (s_preamble_test[j] + (unsigned char)m - s_syms[searchIdx+j]) % m;
                s_ws[t * m + (int)counterIdx] += 1;
            }
        }

        // Wait for the workspace to be complete
        __syncthreads();

        // Write our workspace (which is the matches matrix for this preamble)
        // back to global memory; make sure to skip 
        for (int t = threadIdx.x; t < matchesSzPerPreamble; t += blockDim.x)
        {
            matches[outputBlkGlobalOffset + i * matchesSzPerPreamble + t] = s_ws[t];
        }

        // Before going to the next preamble, make sure everyone is done
        __syncthreads();

    }

}

/*
Where the above kernel assumes that all signals are the same modulation order m,
here we input an array d_m that specifies the order for each row in d_syms.

In order to support this while comparing arbitrary preambles for BPSK/QPSK,
we can no longer store the output matches into a global array, as the 
dimensions of the output matches matrix will be different for BPSK and QPSK respectively.
Instead, the max value of the matches should be taken within this kernel itself.

TODO: complete this.
*/



// extern "C" __global__
// void compareIntegerPreambles_BQ(
//     const unsigned char *d_m,
//     const unsigned char *d_syms,
//     const int numSignals,
//     const int symsLength, 
//     const int searchStart,
//     const int searchEnd,
//     const unsigned char *d_preamblesB,
//     const int numPreamblesB,
//     const int *preambleLengthsB,
//     const unsigned char *d_preamblesQ,
//     const int numPreamblesQ,
//     const int *preambleLengthsQ,
//     unsigned int *matches
// ){
//     // Exit if the block number is more than the number of signals
//     if (blockIdx.x >= numSignals)
//         return;

//     // Read the modulation order for the row
//     int m = (int)d_m[blockIdx.x];

//     // Create temporary pointers/values
//     const int numPreambles;
//     const int *preambleLengths;
//     const unsigned char *d_preambles;
//     if (m == 2)
//     {
//         numPreambles = numPreamblesB;
//         preambleLengths = preambleLengthsB;
//         d_preambles = d_preamblesB;
//     }
//     else if (m == 4)
//     {
//         numPreambles = numPreamblesQ;
//         preambleLengths = preambleLengthsQ;
//         d_preambles = d_preamblesQ;
//     }

//     // Allocate shared memory to read in the signal (one row) and preambles
//     extern __shared__ double s[];
//     int *s_preambleLengths = (int*)s; // (numPreambles) ints

//     // Read in the preamble lengths first
//     for (int t = threadIdx.x; t < numPreambles; t += blockDim.x){
//         s_preambleLengths[t] = preambleLengths[t];
//     }
//     // Wait for the lengths to be read in
//     __syncthreads();

//     // Now check the total length required for all preambles
//     int totalPreambleLength = 0;
//     for (int i = 0; i < numPreambles; i++)
//         totalPreambleLength += s_preambleLengths[i];

//     // And then we assign the space for the rest of shared memory
//     const int matchesSzPerPreamble = m * (searchEnd - searchStart);
//     const int matchesSzPerSignal = numPreambles * matchesSzPerPreamble;
//     const int outputBlkGlobalOffset = blockIdx.x * matchesSzPerSignal;

//     unsigned int *s_ws = (unsigned int*)&s_preambleLengths[numPreambles]; // (m * (searchEnd-searchStart)) unsigned ints, this is the 'matches' matrix
//     unsigned char *s_syms = (unsigned char*)&s_ws[matchesSzPerPreamble]; // (symsLength) unsigned chars
//     unsigned char *s_preambles = (unsigned char*)&s_syms[symsLength]; // (totalPreambleLength) unsigned chars
//     /* IMPORTANT:
//     We stack the shared memory in order of int32->uint32->uint8->uint8
//     because CUDA enforces all pointers of a type to be aligned to an address of that type's size.
//     If we moved the uint32 pointer to the last part of the shared memory,
//     the uint8s may lead to a non-32-bit aligned address as the start of the next memory section,
//     and this will cause a misaligned address error during execution 
//     (
//         NOTE: compilation will be successful,
//         and the kernel will actually run without error if you get lucky!)
//     */


//     // Copy these into shared memory
//     int blkGlobalOffset = blockIdx.x * symsLength;
//     for (int t = threadIdx.x; t < symsLength; t += blockDim.x)
//         s_syms[t] = d_syms[blkGlobalOffset + t]; // Copy only this row

//     for (int t = threadIdx.x; t < totalPreambleLength; t += blockDim.x)
//         s_preambles[t] = d_preambles[t]; // Copy the entire concatenated array of preambles
    
//     // Wait for all shared mem copies to complete
//     __syncthreads();
    
//     // Now we process each preamble individually
//     unsigned char *s_preamble_test = s_preambles;
    
//     unsigned char counterIdx;
//     for (int i = 0; i < numPreambles; i++)
//     {
//         if (i > 0) // Increment pointer to the next preamble
//             s_preamble_test = s_preamble_test + s_preambleLengths[i-1];

//         // Each thread tackles a search index, looping until all search indices are complete
//         for (int t = threadIdx.x; t < searchEnd - searchStart; t += blockDim.x)
//         {
//             int searchIdx = searchStart + t;
            
//             // Pre-zero our workspace
//             for (int j = 0; j < m; j++)
//             {
//                 s_ws[t*m + j] = 0;
//             }

//             // Loop over the current preamble
//             for (int j = 0; j < s_preambleLengths[i]; j++)
//             {
//                 // Calculate the proper rotation by adding m to the input signal
//                 // Explicit upcasting
//                 counterIdx = (s_preamble_test[j] + (unsigned char)m - s_syms[searchIdx+j]) % m;
//                 s_ws[t * m + (int)counterIdx] += 1;
//             }
//         }

//         // Wait for the workspace to be complete
//         __syncthreads();

//         // Write our workspace (which is the matches matrix for this preamble)
//         // back to global memory; make sure to skip 
//         for (int t = threadIdx.x; t < matchesSzPerPreamble; t += blockDim.x)
//         {
//             matches[outputBlkGlobalOffset + i * matchesSzPerPreamble + t] = s_ws[t];
//         }

//         // Before going to the next preamble, make sure everyone is done
//         __syncthreads();

//     }

// }



extern "C" __global__
void lockPhase_mapSyms_singleBlkKernel_qpsk(
    const complex<float> *d_x,
    const int xlength,
    const int *d_amble,
    const int amblelength,
    const int searchstart,
    const int searchlength,
    complex<float> *d_reimc,
    unsigned int *d_syms,
    int *d_matches,
    int *d_rotation,
    int *d_matchIdx,
    unsigned char *d_bits, int bitslen)
{
    /* Note that in the batch, the blockIdx is the batch index */
 
    // allocate shared memory
    extern __shared__ double s[];
    
    complex<float> *s_x = (complex<float>*)s; // (xlength) complex floats
    float *s_ws = (float*)&s_x[xlength]; // (workspace length) floats
    /* workspace length >= blockDim.x*/
    
    // reinterpret for later use as well
    complex<int> *s_syms = (complex<int>*)s; // (xlength) complex ints
    
    // for later use, we also point the reused workspace to other things
    int *s_amble = (int*)&s_ws[0]; // (amblelength) ints
    int *s_matches = (int*)&s_amble[amblelength]; // (searchlength) ints
    int *s_rotation = (int*)&s_matches[searchlength]; // (searchlength) ints
    

    // load shared memory
    for (int t = threadIdx.x; t < xlength; t = t + blockDim.x){
        s_x[t] = d_x[t + blockIdx.x*xlength];
    }

    __syncthreads();
    
    // zero the workspace
    s_ws[threadIdx.x] = 0;
    
    // loop over the signal
    complex<float> reimp;
    int widx = threadIdx.x % 4;
    int tidx = threadIdx.x / 4;
    for (int i = tidx; i < xlength; i += blockDim.x / 4)
    {
        reimp = s_x[i] * s_x[i]; // squared
        if (widx == 0) // accumulate 0,0
        {
            s_ws[threadIdx.x] += reimp.real() * reimp.real();
        }
        else if (widx == 3) // accumulate 1,1
        {
            s_ws[threadIdx.x] += reimp.imag() * reimp.imag();    
        }
        else // accumulate 0,1 or 1,0
        {
            s_ws[threadIdx.x] += reimp.real() * reimp.imag();    
        }
    }
    
    __syncthreads();
    
    // gather into 2x2 at the front
    if (threadIdx.x < 4)
    {
        // remember that we can skip the first 2x2 values
        for (int i = threadIdx.x + 4; i < blockDim.x; i += 4)
        {
            s_ws[threadIdx.x] += s_ws[i];
        }
    }
        
    __syncthreads(); // we cannot place syncthreads in a conditional block!
        
    // hence split the conditional into this next section again
    if (threadIdx.x < 4)
    {
        // perform the 2x2 eigen decomposition
        float T = s_ws[0] + s_ws[3]; // trace
        float D = s_ws[0] * s_ws[3] - s_ws[1] * s_ws[2]; // determinant
        
        float p1 = T/2.0;
        float p2 = sqrtf(fmaf(p1, p1, -D));
        
        // 0 and 2 write the first eigenvector
        if (threadIdx.x % 2 == 0)
        {
            // compute the eigenvalue
            float l1 = p1 + p2;
            
            // 0 writes the eigenvalue
            if (threadIdx.x == 0){s_ws[4] = l1;}
            
            // compute the eigenvector
            s_ws[6+threadIdx.x] = (threadIdx.x == 0) ? (l1 - s_ws[3]) : s_ws[2];
        }
        else // 1 and 3 write the second eigenvector
        {
            // compute the eigenvalue
            float l2 = p1 - p2;
            
            // 1 writes the eigenvalue
            if (threadIdx.x == 1){s_ws[5] = l2;}
            
            // compute the eigenvector
            s_ws[6+threadIdx.x] = (threadIdx.x == 1) ? (l2 - s_ws[3]) : s_ws[2];
        }
        
    }
    
    __syncthreads();
    
    // at this point, the shared memory contains
    // s_ws[0:4] = square matrix
    // s_ws[4:6] = eigenvalues
    // s_ws[6:10] = eigenvectors, columnwise, i.e. 6,8 is e1 // 7,9 is e2
    
    // use first thread to calculate svd_metric
    // note that this is positive semi-definite, so eigvals are always positive
    // hence the first eigenval is by definition the larger one (since the sqrt is positive)
    // if (threadIdx.x == 0)
    //{
    //    s_ws[10] = s_ws[5] / s_ws[4];
    //}
    // no dire need to output this, let's ignore for now
    
    // all threads compute the same phase
    float angleCorrection = atan2f(s_ws[8], s_ws[6]);

    // correct the phase in place
    float real, imag;
    sincosf(-angleCorrection/2.0 + 0.78539816340, &imag, &real); // we shift it to pi/4 for gray coding later
    complex<float> e(real, imag);
    for (int i = threadIdx.x; i < xlength; i += blockDim.x)
    {
        s_x[i] = s_x[i] * e;
        
        // write out
        d_reimc[i + blockIdx.x*xlength] = s_x[i]; // okay up to here!
    }
    
    // finally, we interpret the symbols
    int xsign, ysign;
    int *intptr;
    const int rotChain[4] = {2,0,3,1};
    for (int i = threadIdx.x; i < xlength; i += blockDim.x)
    {
        xsign = signbit(s_x[i].real());
        ysign = signbit(s_x[i].imag());
        
        // for our particular gray coding, we flip the 1<->0
        xsign = xsign ^ 1;
        ysign = ysign ^ 1;
        
        // and then we can just combine and write it out
        xsign = (xsign << 1) | ysign;
        
        // we overwrite into the real part
        s_syms[i] = complex<int>(xsign);
    }
    
    // load the amble, reuse the workspace
    for (int t = threadIdx.x; t < amblelength; t += blockDim.x)
    {
        s_amble[t] = d_amble[t];
    }
    __syncthreads();
    
    // then we scan over the search, with rotations
    int si; // search index
    int matches[4];
    int sym;
    
    for (int i = threadIdx.x; i < searchlength; i += blockDim.x)
    {
        si = i + searchstart;
        
        // manual zeroing
        matches[0] = 0;
        matches[1] = 0;
        matches[2] = 0;
        matches[3] = 0;
        
        
        for (int j = 0; j < amblelength; j++)
        {
            // read and move to stack
            sym = s_syms[si + j].real(); // remember we wrote into the real part
            
            for (int r = 0; r < 4; r++)
            {
                if (r != 0){
                    // rotate it
                    sym = rotChain[sym];
                }
                
                // compare the (rotated) symbol to the amble
                matches[r] += ((sym == d_amble[j])? 1 : 0);

            } // end of rotations
        } // end of amble matches accumulation
        
        // get the maximum of the matches and write it out
        int bestRot = 0;
        int bestMatches = matches[0];
        for (int m = 1; m < 4; m++)
        {
            if (matches[m] > bestMatches)
            {
                bestRot = m;
                bestMatches = matches[m];
            }
        }
        s_matches[i] = bestMatches;
        s_rotation[i] = bestRot;
        
    }
    
    __syncthreads(); // must sync before comparisons to find best match
    
    // get the best match
    int finalRot = s_rotation[0];
    int finalMatch = s_matches[0];
    int finalIdx = 0;
    for (int i = 1; i < searchlength; i++)
    {
        if (s_matches[i] > finalMatch)
        {
            finalRot = s_rotation[i];
            finalMatch = s_matches[i];
            finalIdx = i;
        }
    }
    
    // write only the best rotation/match out
    if (threadIdx.x == 0)
    {
        d_matches[blockIdx.x] = finalMatch;
        d_rotation[blockIdx.x] = finalRot;
        d_matchIdx[blockIdx.x] = finalIdx;
    } 
    
    // write the correct rotation back out
    for (int i = threadIdx.x; i < xlength; i += blockDim.x)
    {
        sym = s_syms[i].real();
        for (int r = 0; r < finalRot; r++)
        {
            sym = rotChain[sym];
        }
        // for use later, we overwrite in the shared memory as well
        s_syms[i].real(sym);
        
        // finally write it out to global
        d_syms[i + blockIdx.x*xlength] = sym;
        
    }
    
    __syncthreads(); // must sync as each thread accesses a different bank in the following section
    
    // For QPSK, each symbol (which occupies 32 bits now) is 2 bits (which we are saving unpacked to 2*8=16 bits)
    // Each thread reads 1 bank ie 32 bits, writes 16 bits (coalesced, but not fully optimal)
    unsigned short twobits;
    unsigned short *write_ptr;
    for (int i = threadIdx.x; i < bitslen / 2; i += blockDim.x) // don't forget /2 for QPSK here
    {
        sym = s_syms[i + finalIdx + amblelength].real();
        // twobits = ((sym & 0x2) << 7) | (sym & 0x1); // the second bit only needs to move by 7
        twobits = ((sym & 0x1) << 8) | ((sym & 0x2) >> 1) ; // may be this, depending on endianness?
        write_ptr = (unsigned short*)&d_bits[blockIdx.x*bitslen + i*2]; // cast it to a short
        *write_ptr = twobits; // write the 16 bits
    }
    
    

 
}