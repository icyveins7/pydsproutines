#pragma once
#include <iostream>
#include "ipp.h"
#include <chrono>
#include "ipp_ext.h"

class SampledLinearInterpolator_64f
{
	private:
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
