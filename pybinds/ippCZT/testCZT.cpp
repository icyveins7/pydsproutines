#include "CZT.h"

int main()
{
    int testlen = 110923; // expect next one to be 111132
    int nextfastlen = next_fast_len(testlen);
    printf("nextfastlen from %d = %d\n", testlen, nextfastlen);

    // Test the class itself
    int N = 10;
    Ipp32f f1 = -1000.0f;
    Ipp32f f2 = 1000.0f;
    Ipp32f fstep = 1.0f;
    Ipp32f fs = (Ipp32f)N;

    // IppCZT32fc czt(11, -0.1f, 0.1f, 0.01f, 1.0f);
    IppCZT32fc czt;
    try{
        czt = IppCZT32fc(N, f1, f2, fstep, fs);
    }
    catch(std::exception &e)
    {
        printf("Caught ctor error: %s\n", e.what());
    }
    ippe::vector<Ipp32fc> in(N);
    for (int i = 0; i < N; i++)
    {
        in.at(i).re = i;
        in.at(i).im = i;
        printf("in[%d] = %f, %f\n", i, in[i].re, in[i].im);
    }
    ippe::vector<Ipp32fc> out(czt.m_k);
    printf("output length = %zd\n", out.size());
    printf("m_nfft = %zd\n", czt.m_dft.getLength());

    // validated.
    for (int i = 0; i < czt.m_ww.size(); i++)
    {
        printf("ww[%d] = %f, %f\n", i, czt.m_ww[i].re, czt.m_ww[i].im);
    }
    printf("\n\n");

    // validated.
    for (int i = 0; i < czt.m_aa.size(); i++)
    {
        printf("aa[%d] = %f, %f\n", i, czt.m_aa[i].re, czt.m_aa[i].im);
    }
    printf("\n\n");

    // // validated.
    // for (int i = 0; i < czt.m_fv.size(); i++)
    // {
    //     printf("fv[%d] = %f, %f\n", i, czt.m_fv[i].re, czt.m_fv[i].im);
    // }
    // printf("\n\n");

    try{
        czt.runRaw(in.data(), out.data());
    }
    catch(std::exception &e)
    {
        printf("Caught runRaw error: %s\n", e.what());
    }
    

    // // all correct!
    // for (int i = 0; i < out.size(); i++)
    // {
    //     printf("out[%d]= %f, %f\n", i, out.at(i).re, out.at(i).im);
    // }

    return 0;
}