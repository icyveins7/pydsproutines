#include "SampledLinearInterpolator.h"

void SampledLinearInterpolator_64f::calcGrads()
{
	for (int i = 0; i < grads.size(); i++){
		grads.at(i) = (yy.at(i+1) - yy.at(i)) / T;
	}
}

void SampledLinearInterpolator_64f::lerp(const double *xxq, double *yyq, int anslen)
{
	// some resizing
	divAns.resize(anslen);
	intPart.resize(anslen);
	remPart.resize(anslen);
	indexes.resize(anslen);
	
	// divide first
	ippsDivC_64f(xxq, T, divAns.data(), anslen);
	// modf the whole array
	ippsModf_64f(divAns.data(), intPart.data(), remPart.data(), anslen);
	// convert to integers for indexing
	ippsConvert_64f32s_Sfs(intPart.data(), indexes.data(), anslen, ippRndNear, 0);
	// reuse the intPart which is not needed any more as the gradients vector
	Ipp64f *gradients = intPart.data();
	ippsZero_64f(gradients, anslen); // zero it out
	Ipp32s idx;
	for (int qi = 0; qi < anslen; qi++){
		idx = indexes.at(qi);
		if (idx >=0 && idx < len-1){ // don't index outside, we need to access the next point as well
			gradients[qi] = grads.at(idx); // now simply use the value directly
		}
		yyq[qi] = yy.at(idx); // write the output value as well
	}
	// multiply gradients into the decimal part and add to output
	ippsAddProduct_64f(remPart.data(), gradients, yyq, anslen);
}