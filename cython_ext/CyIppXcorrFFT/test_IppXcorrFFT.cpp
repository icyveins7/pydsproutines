/*
cl /EHsc /O2 test_IppXcorrFFT.cpp ippcore.lib ipps.lib -I"..\..\ipp_ext\include"
*/

#include "IppXcorrFFT.cpp" // this is an ugly method to get the thing to compile for now

int main()
{
    // create some data
    ippe::vector<Ipp32fc> data(1000);
    for (int i = 0; i < 100; i++){
        data[i].re = i; data[i].im = i+1.0f;
    }

    int cutoutlen = 300;
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
    for (int i = 0; i < 10; i++)
    {
        printf("Performing xcorr...\n");
        xcfft.xcorr(data.data(), data.size(), 0, data.size(), 1); // overshoot, but it should write 0s
    }
        

    // for (int i = 0; i < xcfft.m_productpeaks.size(); i++){
    //     printf("Peak %d: %f, fidx %d \n", i, xcfft.m_productpeaks[i], xcfft.m_freqlistinds[i]);
    // }

    printf("Complete\n");

    return 0;
}