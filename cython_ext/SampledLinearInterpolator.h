#pragma once
#include <iostream>
#include "ipp.h"
#include <chrono>
#include "ipp_ext.h"

class SampledLinearInterpolator_64f
{
	protected:
		int len;
		double T;
	
		// Input arrays to interpolate around
		ippe::vector<Ipp64f> xx;
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
		SampledLinearInterpolator_64f(double *x, double *y, int in_len, double in_T)
			: len{in_len}, T{in_T}
		{
			xx.resize(len);
			yy.resize(len);
	 		grads.resize(len-1);
			ippsCopy_64f(x, xx.data(), xx.size());
			ippsCopy_64f(y, yy.data(), yy.size());
			calcGrads(); // call the pre-calculation
		}
		~SampledLinearInterpolator_64f()
		{
		}
		
		// Main calling function
		void lerp(const double *xxq, double *yyq, int anslen);

		
};

class ConstAmpSigLerp_64f : public SampledLinearInterpolator_64f
{
	private:
		double amp;
		double fc;
		ippe::vector<Ipp64f> tmtau;
		ippe::vector<Ipp64f> ampvec;
		ippe::vector<Ipp64f> phasevec;
		
		// Submethods
		void calcCarrierFreq_TauPhase(const double *tau, int anslen, double *phase);
	
	public:
		ConstAmpSigLerp_64f(double *timevec, double *phasevec, int in_len, double in_T,
							double in_amp, double in_fc)
			: SampledLinearInterpolator_64f{timevec, phasevec, in_len, in_T},
			amp{in_amp}, fc{in_fc}
		{
			
		}
		~ConstAmpSigLerp_64f()
		{
		}
		
		// Main calling function
		void propagate(const double *t, const double *tau, const double phi, int anslen, Ipp64fc *x);
		
		
};

