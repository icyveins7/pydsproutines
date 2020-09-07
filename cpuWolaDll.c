#include <math.h>
#include "stdio.h"
#include <stdlib.h>
#include "fftw3.h"
#include <string.h>
#include <ipp.h>

// #include <pthread.h>

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
	
	fftw_complex *thread_y;
	double *thread_f_tap;
	int thread_L;
	int thread_N;
	int thread_Dec;
	int thread_nprimePts;
	
	fftw_complex *thread_fin;
	fftw_complex *thread_fout;

	Ipp64fc *thread_tones;
	
	fftw_complex *thread_out;
};

// declare global thread stuff
struct thread_data thread_data_array[NUM_THREADS];

// test fftw_plans array on stack for threads, works
fftw_plan allplans[NUM_THREADS]; // REMEMBER TO CHECK FFTW PLANS CREATION IN THE ENTRY FUNCTION

unsigned __stdcall threaded_wola(void *pArgs){
// void *threaded_wola(void *pArgs){
    struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	fftw_complex *y = inner_data->thread_y;
	int L = inner_data->thread_L;
	int N = inner_data->thread_N;
	int Dec = inner_data->thread_Dec;
	int nprimePts = inner_data->thread_nprimePts;
	double *f_tap = inner_data->thread_f_tap;

	fftw_complex *fin = inner_data->thread_fin;
	fftw_complex *fout = inner_data->thread_fout;

	Ipp64fc *tones = inner_data->thread_tones;
	
	fftw_complex *out = inner_data->thread_out; // for R2018
	// end of assignments
    
    int nprime, n, a, b; // declare to simulate threads later
    int k;
	int tone_idx;

	// pick point based on thread number

	for (nprime = t_ID; nprime<nprimePts; nprime=nprime+NUM_THREADS){
        n = nprime*Dec;
        for (a = 0; a<N; a++){
            fin[a][0] = 0; // init to 0
            fin[a][1] = 0;
            for (b = 0; b<L/N; b++){
                if (n - (b*N+a) >= 0){
                    fin[a][0] = fin[a][0] + y[n-(b*N+a)][0] * f_tap[b*N+a];
                    fin[a][1] = fin[a][1] + y[n-(b*N+a)][1] * f_tap[b*N+a];
                } // fin is fftw_complex
            }
        }
        fftw_execute(allplans[t_ID]); // this should place them into another fftw_complex fout
        
		
		// // === old code for up to bin overlap of 2 ===
		// if (Dec*2 == N && nprime % 2 != 0){ // only if using overlapping channels, do some phase corrections when nprime is odd
			// for (k=1; k<N; k=k+2){ //  all even k are definitely even in the product anyway
				// fout[k][0] = -fout[k][0];
				// fout[k][1] = -fout[k][1];
			// }
		// }
		
        // memcpy(&out[nprime*N],fout,sizeof(fftw_complex)*N);
		
		// === new code for general bin overlaps ===
		tone_idx = nprime % (N/Dec);
		ippsMul_64fc((Ipp64fc*)fout, &tones[tone_idx * N], (Ipp64fc*)&out[nprime*N], N);
	}
	
	_endthreadex(0);
    return 0;
}

extern DLL_EXPORT int cpuWola(fftw_complex *y, double *f_tap, int fftlen, int Dec, int nprimePts, int L, fftw_complex *out){
// int cpuWola(fftw_complex *y, double *f_tap, int fftlen, int Dec, int nprimePts, int L, fftw_complex *out){
	ippInit();
	Ipp64fc *tones;
	Ipp64f phase = 0;
	Ipp64f rFreq = 0;

	fftw_complex *fin, *fout;
	int i, t;
	
	tones = ippsMalloc_64fc_L(fftlen/Dec * fftlen); // the tone is fftlen elements, and we need fftlen/Dec of them before the phase correction repeats
	
	for (i=0; i<fftlen/Dec; i++){
		rFreq = -(Ipp64f)i * (Ipp64f)Dec / (Ipp64f)fftlen;
		if (rFreq<0){rFreq = rFreq + 1;} // basically other than the first one, all are negative, so shift to the positive equivalent
		ippsTone_64fc(&tones[i*fftlen], fftlen, 1.0, rFreq, &phase, ippAlgHintNone);
	}
	
	fin = fftw_alloc_complex(fftlen*NUM_THREADS);
	fout = fftw_alloc_complex(fftlen*NUM_THREADS);

    allplans[0] = fftw_plan_dft_1d(fftlen, fin, fout, FFTW_BACKWARD, FFTW_ESTIMATE); // FFTW_MEASURE seems to cut execution time by ~10%, but FFTW_ESTIMATE takes ~0.001s whereas MEASURE takes ~0.375s

    
    for (i=1;i<NUM_THREADS;i++){
        allplans[i] = fftw_plan_dft_1d(fftlen, &fin[fftlen*i], &fout[fftlen*i], FFTW_BACKWARD, FFTW_ESTIMATE); // make the other plans, not executing them yet
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
		
		thread_data_array[t].thread_fin = &fin[t*fftlen];
		thread_data_array[t].thread_fout = &fout[t*fftlen];
		
		thread_data_array[t].thread_tones = tones;

		thread_data_array[t].thread_out = out; // for R2018
		
         // pthread_create(&ThreadList[t], &attr, threaded_wola, (void *)&thread_data_array[t]);
		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_wola,(void*)&thread_data_array[t],0,NULL);

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
   printf("All threads closed! \n");

    for (i=0;i<NUM_THREADS;i++){fftw_destroy_plan(allplans[i]);}

    fftw_free(fin);
    fftw_free(fout);
	
	ippsFree(tones);

	return 0;
}
