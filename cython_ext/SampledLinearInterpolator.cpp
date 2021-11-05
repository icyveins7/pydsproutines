#include "SampledLinearInterpolator.h"

void SampledLinearInterpolator_64f::calcGrads()
{
	grads.resize(len-1);
    ippsSub_64f(&yy.at(0), &yy.at(1), grads.data(), grads.size());

    // // deprecated?
	// for (int i = 0; i < grads.size(); i++){
		// grads.at(i) = (yy.at(i+1) - yy.at(i));
	// }
}

// Input yyq is expected to be pre-zeroed (out of bounds query indices will not be written, so will be left as zero)
void SampledLinearInterpolator_64f::lerp(const double *xxq, double *yyq, int anslen, SampledLinearInterpolatorWorkspace_64f *ws)
{
    // == identical to the original, but repoint all workspace vectors ==
    // some resizing
	ws->divAns.resize(anslen);
	ws->intPart.resize(anslen);
	ws->remPart.resize(anslen);
	ws->indexes.resize(anslen);
    
	// divide first
	ippsDivC_64f(xxq, T, ws->divAns.data(), anslen);
	// modf the whole array
	ippsModf_64f(ws->divAns.data(), ws->intPart.data(), ws->remPart.data(), anslen);
	// convert to integers for indexing
	ippsConvert_64f32s_Sfs(ws->intPart.data(), ws->indexes.data(), anslen, ippRndNear, 0);
	// reuse the intPart which is not needed any more as the gradients vector
	Ipp64f *gradients = ws->intPart.data();
	ippsZero_64f(gradients, anslen); // zero it out
	Ipp32s idx;
	for (int qi = 0; qi < anslen; qi++){
		idx = ws->indexes.at(qi);
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
void ConstAmpSigLerp_64f::propagate(const double *t, const double *tau, const double phi, int anslen, Ipp64fc *x, SampledLinearInterpolatorWorkspace_64f *ws)
{
    // some resizes
	tmtau.resize(anslen);
	phasevec.resize(anslen);
	ampvec.resize(anslen);
	
	// first calculate t - tau
	ippsSub_64f((const Ipp64f*)tau, (const Ipp64f*)t, tmtau.data(), anslen); // src2 - src1
	
	// let's keep some markers
	int startIdx = -1;
	int endIdx = -1;
	
	// and write the ampvec
	ippsZero_64f(ampvec.data(), ampvec.size());
	for (int i = 0; i < ampvec.size(); i++){
		// write const amplitude value only for time values within defined range
		if ((tmtau.at(i) >= timevec_start) && (tmtau.at(i) <= timevec_end)){ 
			ampvec.at(i) = amp;
			if (startIdx == -1){ // on the first one, we set the start marker
				startIdx = i;
				endIdx = i; // and also the end marker
			}
			else{
				endIdx = i; // otherwise, we extend the end marker as long as it sees any single sample satisfying the timevec requirement
			}
				
		}
	}
	
	// now lerp the phase (resize will happen internally)
	// but we don't need to do all of it if it's mostly zeros! use the markers
	// ippsZero_64f(phasevec.data(), phasevec.size()); // zero the array first though (or maybe don't need? since amp will be 0..
	lerp(&tmtau.at(startIdx), &phasevec.at(startIdx), endIdx - startIdx + 1, ws);
	
	
	// calculate phasor change due to carrier frequency
	// divAns.resize(anslen); // unnecessary since lerp should have done it
	Ipp64f *carrierPhase = ws->divAns.data(); // re-use since lerp doesn't require it any more
	calcCarrierFreq_TauPhase(tau, anslen, carrierPhase);
	
	// now add the two phases together
	ippsAdd_64f_I(carrierPhase, phasevec.data(), anslen);
	// and the constant phi
	ippsAddC_64f_I(phi, phasevec.data(), anslen);
	
	// and turn it into complex
	ippsPolarToCart_64fc(ampvec.data(), phasevec.data(), x, anslen);

}

void ConstAmpSigLerp_64f::calcCarrierFreq_TauPhase(const double *tau, int anslen, double *phase)
{
	ippsMulC_64f(tau, -IPP_2PI*fc, phase, anslen); // using the 2 pi define, DON'T PUT A 2* IN FRONT
	// ippsMulC_64f(tau, -IPP_PI*fc, phase, anslen); // debugging
	// ippsZero_64f(phase, anslen); // debugging
}


// ===================================================================================================================
void ConstAmpSigLerpBursty_64f::addSignal(ConstAmpSigLerp_64f* sig)
{
	sDict.push_back(sig);
}

void ConstAmpSigLerpBursty_64f::propagate(const double *t, const double *tau, 
						const double *phiArr, const double *tJumpArr, // these should have length == sDict.size
						int anslen, Ipp64fc *x,
						SampledLinearInterpolatorWorkspace_64f *ws)
{
	// note: no error-checking performed here for lengths of phiArr or tJumpArr..
	
	// temporary arrays
	ippe::vector<Ipp64f> tauPlusJump(anslen);
	ippe::vector<Ipp64fc> xtmp(anslen);
	
	// it is expected that x is already zeroed?
	
	// loop over the signals
	for (int i = 0; i < sDict.size(); i++){
		// Add tau to the jump for the burst
		ippsAddC_64f(tau, tJumpArr[i], tauPlusJump.data(), anslen);
		// Propagate the signal
		sDict.at(i)->propagate(t, tauPlusJump.data(), phiArr[i], anslen, xtmp.data(), ws);
		// Add to x (the output)
		ippsAdd_64fc_I(xtmp.data(), x, anslen);
	}
}


// ===================================================================================================================
void ConstAmpSigLerpBurstyMulti_64f::addSignal(ConstAmpSigLerpBursty_64f* sig)
{
	sigs.push_back(sig);
}

void ConstAmpSigLerpBurstyMulti_64f::propagate(
	const double *t, const double *tau, 
	const double *phiArrs, const double *tJumpArrs, int numBursts,
	int anslen, Ipp64fc *x, int numThreads)
{
	// get ready for threads
	std::vector<std::thread> thds(numThreads);
	// for now, just create the workspace here
	SampledLinearInterpolatorWorkspace_64f** wsvec = (SampledLinearInterpolatorWorkspace_64f**)malloc(sizeof(SampledLinearInterpolatorWorkspace_64f*)*numThreads);
	// std::vector<SampledLinearInterpolatorWorkspace_64f> wsvec; // i don't know why this doesn't work
	// temporary workspace vectors
	std::vector<ippe::vector<Ipp64fc>> xtmpvec(numThreads);
	
	for (int i = 0; i < numThreads; i++){
		wsvec[i] = new SampledLinearInterpolatorWorkspace_64f(anslen);
		// wsvec.push_back(SampledLinearInterpolatorWorkspace_64f(anslen));
		xtmpvec.at(i).resize(anslen);
	}
	
	// invoke a thread for each sig
	for (int i = 0; i < sigs.size(); i++){

		// estimate the first index to be calculated
		
		// and the last index
		
		// zero the vector
		ippsZero_64fc(xtmpvec.at(i % numThreads).data(), anslen);

		// at the start, launch 1 thread each
		if (i < numThreads){ 
			thds.at(i % numThreads) = std::thread(&ConstAmpSigLerpBursty_64f::propagate, sigs.at(i),
													t, tau,
													&phiArrs[i*numBursts], &tJumpArrs[i*numBursts],
													anslen, xtmpvec.at(i % numThreads).data(),
													wsvec[i % numThreads]);
										
		}
		else{ // otherwise we must wait for the previous thread to end first
			// thds.at(i % numThreads).join();
			// add the result into the final (no race condition since only the main thread does this
			ippsAdd_64fc_I(xtmpvec.at(i % numThreads).data(), x, anslen);
			// then relaunch here
			thds.at(i % numThreads) = std::thread(&ConstAmpSigLerpBursty_64f::propagate, sigs.at(i),
													t, tau,
													&phiArrs[i*numBursts], &tJumpArrs[i*numBursts],
													anslen, xtmpvec.at(i % numThreads).data(),
													wsvec[i % numThreads]);
			
			
		}
		
	}
	
	// at the very end, wait for all the threads and add them in too
	for (int i = 0; i < numThreads; i++){
		if (thds.at(i).joinable()){ // need this qualifier to not add things that have already been added
			thds.at(i).join();
			ippsAdd_64fc_I(xtmpvec.at(i).data(), x, anslen);
		}
	}
		
		
	// cleanup?
	for (int i = 0; i < numThreads; i++){
		delete wsvec[i];
	}
	free(wsvec);
}


