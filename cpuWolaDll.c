// cl /O2 /LD cpuWolaDll.c ippcore.lib ipps.lib

#include <math.h>
#include "stdio.h"
#include <stdlib.h>
#include "ipp.h"
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>

#define DLL_EXPORT __declspec(dllexport)

#ifdef __cplusplus
extern "C" {
#endif

// definition of thread data
struct thread_data{
	int thread_t_ID;
	int thread_NUMTHREADS;
	
	Ipp32fc *thread_y;
	float *thread_f_tap;
	int thread_L;
	int thread_N;
	int thread_Dec;
	int thread_nprimePts;
	
	// IPP DFT vars
	Ipp8u *thread_pDFTBuffer;
	IppsDFTSpec_C_32fc *thread_pDFTSpec;

	Ipp32fc *thread_out; // for R2018
};


unsigned __stdcall threaded_wola(void *pArgs){
    struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	int NUM_THREADS = inner_data->thread_NUMTHREADS;
	
	Ipp32fc *y = inner_data->thread_y;
	int L = inner_data->thread_L;
	int N = inner_data->thread_N;
	int Dec = inner_data->thread_Dec;
	int nprimePts = inner_data->thread_nprimePts;
	float *f_tap = inner_data->thread_f_tap;

	// IPP DFT vars
	Ipp8u *pDFTBuffer = inner_data->thread_pDFTBuffer;
	IppsDFTSpec_C_32fc *pDFTSpec = inner_data->thread_pDFTSpec;

	Ipp32fc *out = inner_data->thread_out; // for R2018
	// end of assignments
    
    int nprime, n, a, b; // declare to simulate threads later
    int k;
	
	// allocate for FFTs
	Ipp32fc *dft_in = (Ipp32fc*)ippsMalloc_32fc_L(N);
	Ipp32fc *dft_out = (Ipp32fc*)ippsMalloc_32fc_L(N);

	// pick point based on thread number

	for (nprime = t_ID; nprime<nprimePts; nprime=nprime+NUM_THREADS){
        n = nprime*Dec;
		
		ippsZero_32fc(dft_in, N);
		
        for (a = 0; a<N; a++){
            for (b = 0; b<L/N; b++){
                if (n - (b*N+a) >= 0){
					dft_in[a].re = dft_in[a].re + y[n-(b*N+a)].re * f_tap[b*N+a];
					dft_in[a].im = dft_in[a].im + y[n-(b*N+a)].im * f_tap[b*N+a];
                } 
            }
        }
		
		// ippsDFTInv_CToC_32fc(dft_in, dft_out, pDFTSpec, pDFTBuffer); // actually you can write directly to the matlab output in r2018 since it's interleaved
		ippsDFTInv_CToC_32fc(dft_in, (Ipp32fc*)&out[nprime*N], pDFTSpec, pDFTBuffer);
		

        
		if (Dec*2 == N && nprime % 2 != 0){ // only if using overlapping channels, do some phase corrections when nprime is odd
			for (k=1; k<N; k=k+2){ //  all even k are definitely even in the product anyway
				// dft_out[k].real = -dft_out[k].real;
				// dft_out[k].imag = -dft_out[k].imag; // actually you can write directly to the matlab output in r2018 since it's interleaved
				out[nprime*N + k].re = -out[nprime*N + k].re;
				out[nprime*N + k].im = -out[nprime*N + k].im;
			}
		}
		
        // memcpy(&out[nprime*N],fout,sizeof(Ipp32fc)*N); // if you write directly, you won't need to copy it
	}
	
	ippsFree(dft_in);
	ippsFree(dft_out);
	
	_endthreadex(0);
    return 0;
}

/* The gateway function */
extern DLL_EXPORT int cpuWola(Ipp32fc *y, Ipp32f *f_tap, int fftlen, int Dec, int nprimePts, int L, Ipp32fc *out, int NUM_THREADS){
	ippInit();
    // declare variables
    int t; // for loops over threads
    HANDLE *ThreadList = (HANDLE*)malloc(sizeof(HANDLE) * NUM_THREADS);// handles to threads
	struct thread_data *thread_data_array = (struct thread_data *)malloc(sizeof(struct thread_data) * NUM_THREADS);
	
    // ====== ALLOC VARS FOR FFT IN THREADS BEFORE PLANS ====================
	// ===== IPP DFT Allocations =====
	int sizeSpec = 0, sizeInit = 0, sizeBuf = 0;   
	ippsDFTGetSize_C_32fc(fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuf); // this just fills the 3 integers
	/* memory allocation */
	IppsDFTSpec_C_32fc **pDFTSpec = (IppsDFTSpec_C_32fc**)ippMalloc(sizeof(IppsDFTSpec_C_32fc*)*NUM_THREADS);
	Ipp8u **pDFTBuffer = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	Ipp8u **pDFTMemInit = (Ipp8u**)ippMalloc(sizeof(Ipp8u*)*NUM_THREADS);
	for (t = 0; t<NUM_THREADS; t++){ // make one for each thread
		pDFTSpec[t] = (IppsDFTSpec_C_32fc*)ippMalloc(sizeSpec); // this is analogue of the fftw plan
		pDFTBuffer[t] = (Ipp8u*)ippMalloc(sizeBuf);
		pDFTMemInit[t] = (Ipp8u*)ippMalloc(sizeInit);
		ippsDFTInit_C_32fc(fftlen, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,  pDFTSpec[t], pDFTMemInit[t]); // kinda like making the fftw plan?
	}
	// ================================================================
	for (t=0; t<NUM_THREADS; t++){
		thread_data_array[t].thread_t_ID = t;
		thread_data_array[t].thread_NUMTHREADS = NUM_THREADS;
		
		thread_data_array[t].thread_f_tap = f_tap;
		thread_data_array[t].thread_L = L;
		thread_data_array[t].thread_N = fftlen;
		thread_data_array[t].thread_Dec = Dec;
		thread_data_array[t].thread_nprimePts = nprimePts;
		thread_data_array[t].thread_y = y;
		
		thread_data_array[t].thread_pDFTBuffer = pDFTBuffer[t];
		thread_data_array[t].thread_pDFTSpec = pDFTSpec[t];

		thread_data_array[t].thread_out = out; // for R2018
		
        // pthread_create(&ThreadList[t], &attr, threaded_wola, (void *)&thread_data_array[t]);
		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_wola,(void*)&thread_data_array[t],0,NULL);

        printf("Beginning threadID %i..\n",thread_data_array[t].thread_t_ID);
	}
    
    WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);

	// ============== CLEANUP =================
    // close threads
    printf("Closing threads...\n");
    for(t=0;t<NUM_THREADS;t++){
        CloseHandle(ThreadList[t]);
//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
    }
    printf("All threads closed! \n");

	// === FINAL CLEANUP ===
	for (t=0; t<NUM_THREADS; t++){
		ippFree(pDFTSpec[t]);
		ippFree(pDFTBuffer[t]);
		ippFree(pDFTMemInit[t]);
	}
	ippFree(pDFTSpec);
	ippFree(pDFTBuffer);
	ippFree(pDFTMemInit);

	free(ThreadList);
	free(thread_data_array);
}


#ifdef __cplusplus
}
#endif