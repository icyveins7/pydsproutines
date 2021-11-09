#include "SampledLinearInterpolator.h"

void SampledLinearInterpolator_64f::calcGrads()
{
	grads.resize(len-1);
    ippsSub_64f(&yy.at(0), &yy.at(1), grads.data(), grads.size());
}

// Input yyq is expected to be pre-zeroed (out of bounds query indices will not be written, so will be left as zero)
void SampledLinearInterpolator_64f::lerp(const double *xxq, double *yyq, int anslen, SampledLinearInterpolatorWorkspace_64f *ws)
{
    // == identical to the original, but repoint all workspace vectors ==
    // some resizing
	ws->divAns.resize(anslen);
	ws->intPart.resize(anslen);
	ws->remPart.resize(anslen);
//	ws->indexes.resize(anslen);
    
	// divide first
	ippsDivC_64f(xxq, T, ws->divAns.data(), anslen);
	// modf the whole array
	ippsModf_64f(ws->divAns.data(), ws->intPart.data(), ws->remPart.data(), anslen);
	// instead of using indexes, just repoint + recast divAns, since 64f > 32s
	Ipp32s *indexes = (Ipp32s*)ws->divAns.data();
	// convert to integers for indexing
//	ippsConvert_64f32s_Sfs(ws->intPart.data(), ws->indexes.data(), anslen, ippRndNear, 0);
	ippsConvert_64f32s_Sfs(ws->intPart.data(), indexes, anslen, ippRndNear, 0);
	// reuse the intPart which is not needed any more as the gradients vector
	Ipp64f *gradients = ws->intPart.data();
	ippsZero_64f(gradients, anslen); // zero it out
	Ipp32s idx;
	for (int qi = 0; qi < anslen; qi++){
//		idx = ws->indexes.at(qi);
		idx = indexes[qi];
		if (idx >=0 && idx < len-1){ // don't index outside, we need to access the next point as well
			gradients[qi] = grads.at(idx); // now simply use the value directly
			yyq[qi] = yy.at(idx); // write the output value as well, this is also only written if within bounds
		}
		
	}
	// multiply gradients into the decimal part and add to output
	// note that grads were actually 'diffs' not gradients, since we have normalised by T already in the first division
	ippsAddProduct_64f(ws->remPart.data(), gradients, yyq, anslen); 
}

// ===================================================================================================================
// void ConstAmpSigLerp_64f::calc_tmtau(const double *t, const double *tau, int anslen)
// {
	// tmtau.resize(anslen);
	// ippsSub_64f((const Ipp64f*)tau, (const Ipp64f*)t, anslen); // src2 - src1
// }

void ConstAmpSigLerp_64f::propagate(const double *t, const double *tau, const double phi, int anslen, Ipp64fc *x, SampledLinearInterpolatorWorkspace_64f *ws, int startIdx)
{
    // some resizes
	tmtau.resize(anslen);
	ampvec.resize(anslen);
	
	// first calculate t - tau (the reason why the final burstyMulti is slow is because this is calculated in full each time)
	ippsSub_64f((const Ipp64f*)tau, (const Ipp64f*)t, tmtau.data(), anslen); // src2 - src1
	
	// let's keep some markers
	int sigStartIdx = -1;
	int sigEndIdx = -1;
	
	// and write the ampvec
	ippsZero_64f(ampvec.data(), ampvec.size());
	int ampStartIdx = (startIdx >= 0 && startIdx<anslen) ? startIdx : 0; // set to startIdx if in bounds, otherwise start at 0
	for (int i = ampStartIdx; i < ampvec.size(); i++){
		// write const amplitude value only for time values within defined range
		if ((tmtau.at(i) >= timevec_start) && (tmtau.at(i) <= timevec_end)){ 
			ampvec.at(i) = amp;
			if (sigStartIdx == -1){ // if uninitialized, we set the start marker
				sigStartIdx = i;
				sigEndIdx = i; // and also the end marker
			}
			else{
				sigEndIdx = i; // otherwise, we extend the end marker as long as it sees any single sample satisfying the timevec requirement
			}
				
		}
		
		// cut loop early once it is over (assumes causality i.e. later samples cannot appear before earlier samples)
		if (tmtau.at(i) > timevec_end){
		    break;
	    }
	}
    // save the finalIdx internally (this will be useful when iterating over multiple instances of this class,
    // but using only 1 timevec (then we don't need to pass through the array multiple times!
    finalIdx = sigEndIdx;
    int siglen = sigEndIdx - sigStartIdx + 1;
	
	// TODO: add some error-checking here, but it works otherwise
	
	// now lerp the phase (resize will happen internally)
	// but we don't need to do all of it if it's mostly zeros! use the markers
	// ippsZero_64f(phasevec.data(), phasevec.size()); // zero the array first though (or maybe don't need? since amp will be 0..
	phasevec.resize(siglen); // moved the resize here, to the shorter length
    lerp(&tmtau.at(sigStartIdx),
		phasevec.data(), // and so now write at index 0
		siglen,
		ws);
	
	
	// calculate phasor change due to carrier frequency
    // lerp has resized divAns to size siglen
	Ipp64f *carrierPhase = ws->divAns.data(); // re-use since lerp doesn't require it any more
	calcCarrierFreq_TauPhase(&tau[sigStartIdx], siglen, carrierPhase);
	
	// now add the two phases together
	ippsAdd_64f_I(carrierPhase, phasevec.data(), siglen);
	// and the constant phi
	ippsAddC_64f_I(phi, phasevec.data(), siglen);
	
	// and turn it into complex (but only write to the viable indices)
//	ippsPolarToCart_64fc(ampvec.data(), phasevec.data(), x, anslen);
    ippsPolarToCart_64fc(&ampvec.at(sigStartIdx),
						phasevec.data(), // phasevec was filled from idx 0
						&x[sigStartIdx],
						siglen); // this assumes x is pre-zeroed
}

void ConstAmpSigLerp_64f::calcCarrierFreq_TauPhase(const double *tau, int anslen, double *phase)
{
	ippsMulC_64f(tau, -IPP_2PI*fc, phase, anslen); // using the 2 pi define, DON'T PUT A 2* IN FRONT
}


// ===================================================================================================================
void ConstAmpSigLerpBursty_64f::addSignal(ConstAmpSigLerp_64f* sig)
{
	sDict.push_back(sig);
}

int ConstAmpSigLerpBursty_64f::propagate(const double *t, const double *tau, 
						const double *phiArr, const double *tJumpArr, // these should have length == sDict.size
						int anslen, Ipp64fc *x,
						SampledLinearInterpolatorWorkspace_64f *ws, int startIdx)
{
	// note: no error-checking performed here for lengths of phiArr or tJumpArr..
	
	// temporary arrays
	ippe::vector<Ipp64f> tauPlusJump(anslen);
	// ippe::vector<Ipp64fc> xtmp(anslen);
	
	// it is expected that x is already zeroed?
	
	// loop over the signals
	int nextStartIdx;
	int outOfBoundsErrs = 0;
	for (int i = 0; i < sDict.size(); i++){
		// Add tau to the jump for the burst
		ippsAddC_64f(tau, tJumpArr[i], tauPlusJump.data(), anslen);
		// Propagate the signal
		// sDict.at(i)->propagate(t, tauPlusJump.data(), phiArr[i], anslen, xtmp.data(), ws, startIdx);
		// // Add to x (the output)
		// ippsAdd_64fc_I(xtmp.data(), x, anslen);
		
		// Now that the internal propagator checks for range and writes directly, do not perform the final add
		sDict.at(i)->propagate(t, tauPlusJump.data(), phiArr[i], anslen, x, ws, startIdx);
		
		// Before continuing, update the startIdx to the finalIdx of the current signal, so we iterate forwards only
		nextStartIdx = sDict.at(i)->getFinalIdx();
		// However, if it is not in range (ie errored), we revert to the original startIdx
		startIdx = (nextStartIdx >= -1 && nextStartIdx < anslen) ? nextStartIdx : startIdx;
	}
	
	// update internal finalIdx similarly
	finalIdx = sDict.back()->getFinalIdx();
	
	return outOfBoundsErrs; // TODO: make this accumulate errors appropriately
}


// ===================================================================================================================
void ConstAmpSigLerpBurstyMulti_64f::addSignal(ConstAmpSigLerpBursty_64f* sig)
{
	sigs.push_back(sig);
}

void ConstAmpSigLerpBurstyMulti_64f::threadPropagate(
	const double *t, const double *tau,
	const double *phiArrs, const double *tJumpArrs, int numBursts,
	int anslen, Ipp64fc *xtmpvec, int numThreads, int threadIdx)
{
	// make your own workspace for each thread
	SampledLinearInterpolatorWorkspace_64f ws; // default to not allocating at start, to prevent excessively large memory blocks
	
	// some vars
	int startIdx;
    double T = t[1] - t[0]; // estimate sample period
	
	// loop over the signals with a stride of numThreads
	for (int i = threadIdx; i < sigs.size(); i = i + numThreads)
	{
		// conservative estimate of startIdxs
		// note that this makes the following fundamental assumptions:
		// 1) 't' vector starts at 0
		// 2) 't-tau' is strictly increasing; this implies that the first index found will always be the first index, even if tau is decreasing
		startIdx = (int)(tJumpArrs[i*numBursts] / T);
		if (startIdx <= -1){ startIdx = -1; } // this is the default minimum

		// run the propagator for the signal
		sigs.at(i)->propagate(t, tau, 
							&phiArrs[i*numBursts], &tJumpArrs[i*numBursts],
							anslen, xtmpvec,
							&ws, startIdx);
	}

}

int ConstAmpSigLerpBurstyMulti_64f::propagate(
	const double *t, const double *tau, 
	const double *phiArrs, const double *tJumpArrs, int numBursts,
	int anslen, Ipp64fc *x, int numThreads)
{
    int err = 0;
    
    // // non-threaded debugging
    // try{
        // for (int thIdx = 0; thIdx < numThreads; thIdx++){
            // threadPropagate(t, tau, phiArrs, tJumpArrs, numBursts, anslen, x, numThreads, thIdx);
        // }
    // }
    
    try{
        // get ready for threads
        std::vector<std::thread> thds;
        thds.resize(numThreads);
        
        // // temporary workspace vectors
        // std::vector<ippe::vector<Ipp64fc>> xtmpvec(numThreads);
        
        // for (int i = 0; i < numThreads; i++){
            // xtmpvec.at(i).resize(anslen);
            // // zero the vector
            // ippsZero_64fc(xtmpvec.at(i).data(), anslen);
        // }
        
        // start threads
        for (int thIdx = 0; thIdx < numThreads; thIdx++){
            // thds.at(thIdx) = std::thread(&ConstAmpSigLerpBurstyMulti_64f::threadPropagate,
                                    // this,
                                    // t, tau,
                                    // phiArrs, tJumpArrs, numBursts,
                                    // anslen, xtmpvec.at(thIdx).data(), numThreads, thIdx);
            
            // in theory, if well separated, then can write directly to the output
            // (since all the writes are performed only in viable indices, so no race conditions,
            // BUT THIS IS RISKY)
            thds.at(thIdx) = std::thread(&ConstAmpSigLerpBurstyMulti_64f::threadPropagate,
                                    this,
                                    t, tau,
                                    phiArrs, tJumpArrs, numBursts,
                                    anslen, x, numThreads, thIdx);
            
        }
        
        // join threads and sum
        for (int thIdx = 0; thIdx < numThreads; thIdx++){
            thds.at(thIdx).join();
            
            // // if doing the RISKY thing above, ignore this
            // ippsAdd_64fc_I(xtmpvec.at(thIdx).data(), x, anslen);
        }
    }
    catch(...)
    {
        FILE *fp = fopen("D:\\gitrepos\\pydsproutines\\wtf.log","w");
        char str[] = "wtf error?";
        fwrite(str, 1, sizeof(str), fp);
        fclose(fp);
        
        err = 1;
    }
    
    return err;
}


