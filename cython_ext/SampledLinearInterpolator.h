#pragma once
#include <iostream>
#include "ipp.h"
#include <chrono>
#include "ipp_ext.h"
#include <vector>
#include <thread>

// Workspace class, for external management, so that each thread can re-use this
class SampledLinearInterpolatorWorkspace_64f
{
    public:
        SampledLinearInterpolatorWorkspace_64f(int size)
        {
            // resize at start
            divAns.resize(size);
            intPart.resize(size);
            remPart.resize(size);
            indexes.resize(size);
        }
        ~SampledLinearInterpolatorWorkspace_64f()
        {
        }
        
        // Vectors
        ippe::vector<Ipp64f> divAns;
        ippe::vector<Ipp64f> intPart;
        ippe::vector<Ipp64f> remPart;
        ippe::vector<Ipp32s> indexes;


};

// This class expects that the original array x starts at 0, and is spaced at 'T' every sample
class SampledLinearInterpolator_64f
{
	protected:
		int len;
		double T;
	
		// Input arrays to interpolate around
		ippe::vector<Ipp64f> yy;
		ippe::vector<Ipp64f> grads;
		// Workspace vectors
		ippe::vector<Ipp64f> divAns;
		ippe::vector<Ipp64f> intPart;
		ippe::vector<Ipp64f> remPart;
		ippe::vector<Ipp32s> indexes;
		
		// Pre-calculation
		void calcGrads();
		
	public:
		SampledLinearInterpolator_64f(double *y, int in_len, double in_T)
			: len{in_len}, T{in_T}
		{
			yy.resize(len);
			ippsCopy_64f(y, yy.data(), yy.size());
			
			calcGrads(); // call the pre-calculation
		}
		~SampledLinearInterpolator_64f()
		{
		}
		
		// Main calling function
		void lerp(const double *xxq, double *yyq, int anslen, SampledLinearInterpolatorWorkspace_64f *ws); // when using external workspace, if this works well then just remove the other one?

		
};

class ConstAmpSigLerp_64f : public SampledLinearInterpolator_64f
{
	private:
		double amp;
		double fc;
		double timevec_start, timevec_end;
		ippe::vector<Ipp64f> tmtau;
		ippe::vector<Ipp64f> ampvec;
		ippe::vector<Ipp64f> phasevec;
		int finalIdx;
		
		// Submethods
		void calcCarrierFreq_TauPhase(const double *tau, int anslen, double *phase);
	
	public:
		ConstAmpSigLerp_64f(double in_timevec_start, double in_timevec_end, double *phasevec, int in_len, double in_T,
							double in_amp, double in_fc)
			: SampledLinearInterpolator_64f{phasevec, in_len, in_T},
			amp{in_amp}, fc{in_fc}, timevec_start{in_timevec_start}, timevec_end{in_timevec_end}
		{
		}
		~ConstAmpSigLerp_64f()
		{
		}
		
		// Main calling function
		void propagate(const double *t, const double *tau, const double phi, int anslen,
						Ipp64fc *x,
						SampledLinearInterpolatorWorkspace_64f *ws, // again, for external management
						int startIdx=-1); // default startIdx (helps for long timevec to specify it)
						
		// Getter for finalIdx
		int getFinalIdx() { return finalIdx; }
		
		
};

class ConstAmpSigLerpBursty_64f
{
	private:
		std::vector<ConstAmpSigLerp_64f*> sDict; // contains the pointer to each burst
		int finalIdx;
		
	public:
		ConstAmpSigLerpBursty_64f()
		{
		}
		~ConstAmpSigLerpBursty_64f()
		{
		}
		
		// Method to add signals
		void addSignal(ConstAmpSigLerp_64f* sig);
		// Method to propagate signals
		void propagate(const double *t, const double *tau, 
						const double *phiArr, const double *tJumpArr, // these should have length == sDict.size
						int anslen, Ipp64fc *x,
						SampledLinearInterpolatorWorkspace_64f *ws, // again, for external management
						int startIdx=-1); // similar to above, helps to specify start
		
};

// Yes the names are getting long, but i'm keeping the convention here for clarity..
class ConstAmpSigLerpBurstyMulti_64f
{
	private:
		std::vector<ConstAmpSigLerpBursty_64f*> sigs;
		
	public:
		ConstAmpSigLerpBurstyMulti_64f()
		{
		}
		~ConstAmpSigLerpBurstyMulti_64f()
		{
		}
		
		// Method to add signals
		void addSignal(ConstAmpSigLerpBursty_64f* sig);
		// Method to propagate
		void propagate(const double *t, const double *tau, 
						const double *phiArrs, const double *tJumpArrs, int numBursts, // these should have length == sDict.size
						int anslen, Ipp64fc *x,	int numThreads);
	
};
