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
	int thread_numChans;
	int thread_chanLen;
	
	Ipp32f *thread_out;
};

// declare global thread stuff
struct thread_data thread_data_array[NUM_THREADS];


unsigned __stdcall threaded_perchan_minMaxScale(void *pArgs){
// void *threaded_wola(void *pArgs){
    struct thread_data *inner_data;
	inner_data = (struct thread_data *)pArgs;
	
	int t_ID = inner_data->thread_t_ID;
	int numChans = inner_data->thread_numChans;
	int chanLen = inner_data->thread_chanLen;
	Ipp32fc *y = inner_data->thread_y;

	Ipp32f *out = inner_data->thread_out; // for R2018
	// end of assignments
    
    int chanIdx, i; // declare to simulate threads later

	Ipp32fc *y_chan; // holder pointer to the current channel
	Ipp32f *chanMagn = (Ipp32f*)ippsMalloc_32f_L(chanLen);
	Ipp32f *chanDiff = (Ipp32f*)ippsMalloc_32f_L(chanLen);
	Ipp32f chanAbsMax, chanAbsMin;
	Ipp32f chanRange;
	
	// pick point based on thread number
	for (chanIdx = t_ID; chanIdx<numChans; chanIdx=chanIdx+NUM_THREADS){
		y_chan = &y[chanIdx*chanLen];
		
		// calculate the abs of the channel
		ippsMagnitude_32fc(y_chan, chanMagn, chanLen);
		
		// find the min and the max of the abs
		ippsMax_32f(chanMagn, chanLen, &chanAbsMax);
		ippsMin_32f(chanMagn, chanLen, &chanAbsMin);
		chanRange = chanAbsMax - chanAbsMin;
		
		// now first minus the min
		ippsSubC_32f(chanMagn, chanAbsMin, chanDiff, chanLen);
		
		// then divide by the range and save it directly
		ippsDivC_32f(chanDiff, chanRange, &out[chanIdx*chanLen], chanLen);
	}
	
	ippsFree(chanMagn);
	ippsFree(chanDiff);
	
	_endthreadex(0);
    return 0;
}

// so now y is chans * nPts
// and out should be chans * minichans * nprimePts
extern DLL_EXPORT int multiChan_minMaxScaler_32fc(Ipp32fc *y, int numChans, int chanLen, Ipp32f *out){
	ippInit();

	int t;
	
    HANDLE ThreadList[NUM_THREADS]; // handles to threads
	    // // stuff for pthreads
	    // pthread_t ThreadList[NUM_THREADS];
	    // pthread_attr_t attr;
	    // pthread_attr_init(&attr);
	    // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    for (t=0; t<NUM_THREADS; t++){
		thread_data_array[t].thread_t_ID = t;
		
		thread_data_array[t].thread_y = y;
		thread_data_array[t].thread_numChans = numChans;
		thread_data_array[t].thread_chanLen = chanLen;
		
		thread_data_array[t].thread_out = out; // for R2018
		
         // pthread_create(&ThreadList[t], &attr, threaded_wola, (void *)&thread_data_array[t]);
		ThreadList[t] = (HANDLE)_beginthreadex(NULL,0,&threaded_perchan_minMaxScale,(void*)&thread_data_array[t],0,NULL);

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

	return 0;
}
