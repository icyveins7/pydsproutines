#include "CZT.h"

int main()
{
    int testlen = 110923; // expect next one to be 111132
    int nextfastlen = next_fast_len(testlen);
    printf("nextfastlen from %d = %d\n", testlen, nextfastlen);

    // Test the class itself
    IppCZT32fc czt(11, -0.1f, 0.1f, 0.01f, 1.0f);
    ippe::vector<Ipp32fc> in(11);
    for (int i = 0; i < 11; i++)
    {
        in.at(i).re = i;
        in.at(i).im = i;
        printf("in[%d] = %f, %f\n", i, in[i].re, in[i].im);
    }
    ippe::vector<Ipp32fc> out(czt.m_k);
    printf("output length = %zd\n", out.size());
    printf("m_nfft = %d\n", czt.m_nfft);

    for (int i = 0; i < czt.m_ww.size(); i++)
    {
        printf("ww[%d] = %f, %f\n", i, czt.m_ww[i].re, czt.m_ww[i].im);
    }
    printf("\n\n");

    for (int i = 0; i < czt.m_aa.size(); i++)
    {
        printf("aa[%d] = %f, %f\n", i, czt.m_aa[i].re, czt.m_aa[i].im);
    }
    printf("\n\n");

    for (int i = 0; i < czt.m_fv.size(); i++)
    {
        printf("fv[%d] = %f, %f\n", i, czt.m_fv[i].re, czt.m_fv[i].im);
    }
    printf("\n\n");


    czt.run(in.data(), out.data());

    for (int i = 0; i < out.size(); i++)
    {
        printf("out[%d]= %f, %f\n", i, out.at(i).re, out.at(i).im);
    }

    return 0;
}