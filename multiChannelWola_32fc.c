#include <math.h>
#include "stdio.h"
#include <stdlib.h>
#include <string.h>
#include <ipp.h>


#include <windows.h>
#include <process.h>

#define NUM_THREADS 22
#define DLL_EXPORT __declspec(dllexport)

#ifdef __cplusplus
extern "C" {
#endif

// definition of thread data
struct thread_data{
	int thread_t_ID;
	
	Ipp32fc *thread_y;
	Ipp32f *thread_f_tap;
	int thread_L;
	int thread_N;
	int thread_Dec;
	int thread_nprimePts;
	int thread_numChans;
	int thread_chanLen;
	
	Ipp32fc *thread_tones;
	
	// IPP DFT vars
	Ipp8u *thread_pDFTBuffer;
	IppsDFTSpec_C_32fc *thread_pDFTSpec;
	
	Ipp32fc *thread_out;
};

// declare global thread stuff
struct thread_data thread_data_array[NUM_THREADS];


unsigned __stdcall threaded_perchannel_wola(void *pArgs){
// void *threaded_wola(void *pArgs){
    struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	Ipp32fc *y = inner_data->thread_y;
	int L = inner_data->thread_L;
	int N = inner_data->thread_N;
	int Dec = inner_data->thread_Dec;
	int numChans = inner_data->thread_numChans;
	int chanLen = inner_data->thread_chanLen;
	int nprimePts = inner_data->thread_nprimePts;
	Ipp32f *f_tap = inner_data->thread_f_tap;

	Ipp32fc *tones = inner_data->thread_tones;
	
	// IPP DFT vars
	Ipp8u *pDFTBuffer = inner_data->thread_pDFTBuffer;
	IppsDFTSpec_C_32fc *pDFTSpec = inner_data->thread_pDFTSpec;
	
	Ipp32fc *out = inner_data->thread_out; // for R2018
	// end of assignments
    
    int nprime, n, a, b, chanIdx; // declare to simulate threads later
    int k;
	int tone_idx;
	Ipp32fc *y_chan; // holder pointer to the current channel
	
	// allocate for FFTs
	Ipp32fc *dft_in = (Ipp32fc*)ippsMalloc_32fc_L(N);
	Ipp32fc *dft_out = (Ipp32fc*)ippsMalloc_32fc_L(N);

	// pick point based on thread number

	for (chanIdx = t_ID; chanIdx<numChans; chanIdx=chanIdx+NUM_THREADS){
		y_chan = &y[chanIdx*chanLen];
		// printf("Thread: %i, chanIdx: %i, first element of chan: %g %g \n", t_ID, chanIdx, y_chan[0].re, y_chan[0].im);
		for (nprime = 0; nprime<nprimePts; nprime++){
			n = nprime*Dec;
			ippsZero_32fc(dft_in, N);
			
			for (a = 0; a<N; a++){
				for (b = 0; b<L/N; b++){
					if (n - (b*N+a) >= 0){
						dft_in[a].re = dft_in[a].re + y_chan[n-(b*N+a)].re * f_tap[b*N+a];
						dft_in[a].im = dft_in[a].im + y_chan[n-(b*N+a)].im * f_tap[b*N+a];
					} 
				}
			}

			ippsDFTInv_CToC_32fc(dft_in, dft_out, pDFTSpec, pDFTBuffer);

			// === new code for general bin overlaps ===
			tone_idx = nprime % (N/Dec);
			ippsMul_32fc(dft_out, &tones[tone_idx * N], (Ipp32fc*)&out[nprimePts*chanIdx*N + nprime*N], N);
		}
	}
	
	ippsFree(dft_in);
	ippsFree(dft_out);
	
	_endthreadex(0);
    return 0;
}

// so now y is chans * nPts
// and out should be chans * minichans * nprimePts
extern DLL_EXPORT int multiChanWOLA(Ipp32fc *y, Ipp32f *f_tap, int fftlen, int Dec, int nprimePts, int L, int numChans, int chanLen, Ipp32fc *out){
	ippInit();
	Ipp32fc *tones;
	Ipp32f phase = 0;
	Ipp32f rFreq = 0;

	int i, t;
	
	tones = ippsMalloc_32fc_L(fftlen/Dec * fftlen); // the tone is fftlen elements, and we need fftlen/Dec of them before the phase correction repeats
	
	for (i=0; i<fftlen/Dec; i++){
		rFreq = -(Ipp64f)i * (Ipp64f)Dec / (Ipp64f)fftlen;
		if (rFreq<0){rFreq = rFreq + 1;} // basically other than the first one, all are negative, so shift to the positive equivalent
		ippsTone_32fc(&tones[i*fftlen], fftlen, 1.0, rFreq, &phase, ippAlgHintNone);
	}
	
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

    
   HANDLE ThreadList[NUM_THREADS]; // handles to threads
	    // // stuff for pthreads
	    // pthread_t ThreadList[NUM_THREADS];
	    // pthread_attr_t attr;
	    // pthread_attr_init(&attr);
	    // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    for (t=0; t<NUM_THREADS; t++){
		thread_data_array[t].thread_t_ID = t;
		
		thread_data_array[t].thread_f_tap = f_tap;
		thread_data_array[t].thread_L = L;
		thread_data_array[t].thread_N = fftlen;
		thread_data_array[t].thread_Dec = Dec;
		thread_data_array[t].thread_nprimePts = nprimePts;
		thread_data_array[t].thread_y = y;
		thread_data_array[t].thread_numChans = numChans;
		thread_data_array[t].thread_chanLen = chanLen;
		
		thread_data_array[t].thread_tones = tones;

		thread_data_array[t].thread_pDFTBuffer = pDFTBuffer[t];
		thread_data_array[t].thread_pDFTSpec = pDFTSpec[t];
		
		thread_data_array[t].thread_out = out; // for R2018
		
         // pthread_create(&ThreadList[t], &attr, threaded_wola, (void *)&thread_data_array[t]);
		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_perchannel_wola,(void*)&thread_data_array[t],0,NULL);

        // printf("Beginning threadID %i..\n",thread_data_array[t].thread_t_ID);
	}

    // for (i = 0; i < NUM_THREADS; i++) {
	    // if(pthread_join(ThreadList[i], NULL)) { // this essentially waits for all above threads
			    // fprintf(stderr, "Error joining threadn");
			    // return 2;
	    // }
    // }
    
   WaitForMultipleObjects(NUM_THREADS,ThreadList,1,INFINITE);

	// ============== CLEANUP =================
	// close threads
	// printf("Closing threads...\n");
	for(t=0;t<NUM_THREADS;t++){
	   CloseHandle(ThreadList[t]);
	//         printf("Closing threadID %i.. %i\n",(int)ThreadIDList[t],WaitForThread[t]);
	}
	// printf("All threads closed! \n");

    // === FINAL CLEANUP ===
	for (t=0; t<NUM_THREADS; t++){
		ippFree(pDFTSpec[t]);
		ippFree(pDFTBuffer[t]);
		ippFree(pDFTMemInit[t]);
	}
	ippFree(pDFTSpec);
	ippFree(pDFTBuffer);
	ippFree(pDFTMemInit);
	
	ippsFree(tones);

	return 0;
}
