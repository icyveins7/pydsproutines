#include <cupy/complex.cuh>

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