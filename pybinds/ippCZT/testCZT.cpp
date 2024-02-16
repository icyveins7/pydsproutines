/*
This entire test file appears to crash semi-consistently (more often on Release, less often on Debug, never when -fsanitize=address?)
but only on AMD chips.

Running it on an Intel chip appears to cause zero crashes. It is too difficult to pin down if this is a bug or not.

*/

#include "CZT.h"
#include <vector>

int main()
{
    //int testlen = 101; // expect next one to be 111132
    //int nextfastlen = next_fast_len(testlen);
    //printf("nextfastlen from %d = %d\n", testlen, nextfastlen);

    int numLoops = 100;
    for (int l = 0; l < numLoops; l++)
    {
        // Test the class itself
        int N = 10; // testlen;
        Ipp32f f1 = -1000.0f;
        Ipp32f f2 = 1000.0f;
        Ipp32f fstep = 1.0f;
        Ipp32f fs = 10000.0f;

        // IppCZT32fc czt(11, -0.1f, 0.1f, 0.01f, 1.0f);
        IppCZT32fc czt(N + 10, f1, f2, fstep, fs);
        // print for this first
        printf("m_ws (%4zd): %p->%p\n", czt.m_ws.size(), czt.m_ws.data(), &czt.m_ws.back());
        printf("m_ws2(%4zd): %p->%p\n", czt.m_ws.size(), czt.m_ws2.data(), &czt.m_ws2.back());
        printf("%p\n%p\n", czt.m_dft.getDFTSpec().data(), czt.m_dft.getDFTBuf().data());
        printf("%zd\n%zd\n", czt.m_dft.getDFTSpec().size(), czt.m_dft.getDFTBuf().size());
        printf("%zd\n%zd\n", czt.m_dft.getDFTSpec().capacity(), czt.m_dft.getDFTBuf().capacity());

        czt = IppCZT32fc(N, f1, f2, fstep, fs);
        // then for this
        printf("m_ws (%4zd): %p\n", czt.m_ws.size(), czt.m_ws.data());
        printf("m_ws2(%4zd): %p\n", czt.m_ws.size(), czt.m_ws2.data());
        printf("%p\n%p\n", czt.m_dft.getDFTSpec().data(), czt.m_dft.getDFTBuf().data());
        printf("%zd\n%zd\n", czt.m_dft.getDFTSpec().size(), czt.m_dft.getDFTBuf().size());
        printf("%zd\n%zd\n", czt.m_dft.getDFTSpec().capacity(), czt.m_dft.getDFTBuf().capacity());


        //IppCZT32fc czt(N, f1, f2, fstep, fs);
        // IppCZT32fc czt;
        // try{
        //     czt = IppCZT32fc(N, f1, f2, fstep, fs);
        // }
        // catch(std::exception &e)
        // {
        //     printf("Caught ctor error: %s\n", e.what());
        //     return -1;
        // }
        ippe::vector<Ipp32fc> in(N);
        for (int i = 0; i < N; i++)
        {
            in.at(i).re = i;
            in.at(i).im = i;
            // printf("in[%d] = %f, %f\n", i, in[i].re, in[i].im);
        }
        ippe::vector<Ipp32fc> out(czt.m_k);
        printf("output length = %zd\n", out.size());
        printf("m_nfft = %zd\n", czt.m_dft.getLength());

        // // validated.
        // for (int i = 0; i < czt.m_ww.size(); i++)
        // {
        //     printf("ww[%d] = %f, %f\n", i, czt.m_ww[i].re, czt.m_ww[i].im);
        // }
        // printf("\n\n");

        // // validated.
        // for (int i = 0; i < czt.m_aa.size(); i++)
        // {
        //     printf("aa[%d] = %f, %f\n", i, czt.m_aa[i].re, czt.m_aa[i].im);
        // }
        // printf("\n\n");

        // // validated.
        // for (int i = 0; i < czt.m_fv.size(); i++)
        // {
        //     printf("fv[%d] = %f, %f\n", i, czt.m_fv[i].re, czt.m_fv[i].im);
        // }
        // printf("\n\n");

        try {
            czt.runRaw(in.data(), out.data());
        }
        catch (std::exception& e)
        {
            printf("Caught runRaw error: %s\n", e.what());
        }


        //// Testing a vector of objects
        //std::vector<IppCZT32fc> cztvec;
        //cztvec.reserve(1);
        //cztvec.emplace_back(N, f1, f2, fstep, fs);
        //cztvec.push_back(czt);
        //cztvec.push_back(czt);

        // // all correct!
        // for (int i = 0; i < out.size(); i++)
        // {
        //     printf("out[%d]= %f, %f\n", i, out.at(i).re, out.at(i).im);
        // }

    }

    printf("Ok\n");
    return 0;
}