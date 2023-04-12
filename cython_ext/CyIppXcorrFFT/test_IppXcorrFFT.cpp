/*
cl /EHsc /O2 test_IppXcorrFFT.cpp IppXcorrFFT.cpp ippcore.lib ipps.lib -I"..\..\ipp_ext\include"
*/

#include "IppXcorrFFT.h"

int main()
{
    // create some data
    ippe::vector<Ipp32fc> data(100);
    for (int i = 0; i < data.size(); i++){
        data[i].re = i; data[i].im = i+1.0f;
    }

    int cutoutlen = 30;
    ippe::vector<Ipp32fc> cutout(cutoutlen);
    // copy from a place in data
    int c = 20;
    for (int i = 0; i < cutoutlen; i++){
        cutout[i].re = data[c+i].re;
        cutout[i].im = data[c+i].im;
    }


    // create xcorr object
    IppXcorrFFT_32fc xcfft(cutout.data(), cutout.size(), 1, true);
    
    // loop arbitrarily many times to see the error
    for (int i = 0; i < 2; i++)
    {
        printf("Performing xcorr...\n");
        xcfft.xcorr(data.data(), data.size(), 0, data.size(), 3); // overshoot, but it should write 0s
    }
        

    for (int i = 0; i < xcfft.m_productpeaks.size(); i++){
        printf("Peak %d: %f, fidx %d \n", i, xcfft.m_productpeaks[i], xcfft.m_freqlistinds[i]);
    }

    printf("Complete\n");

    return 0;
}