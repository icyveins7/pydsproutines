/*
 * Generates a single fast frequency shift.
 *
 * Sample calling structure:
 * cpuTone(len, freq, fs)
 * Inputs: signal length, frequency to shift by, sampling rate
 *
 * Uses IPP to quickly recreate exp(1i*2*pi*freq*(0:len-1)/fs) quickly.
 *
 * gcc -c cpuToneDll.c -fpic -o cpuToneDll.o
 * gcc -shared -o cpuToneDll.so cpuToneDll.o -lippcore -lipps
 *
 * Note that order matters for the above; in particular, linking ippcore/ipps must be done after the object file (.o) inclusion.
 * Speedup for (100M length): 3.39s for numpy, 308ms for this.
*/

#include "ipp.h"

#ifdef __linux__ 
    #define DLL_EXPORT 
#elif _WIN32
    #include <windows.h>
    #include <process.h>
    #define DLL_EXPORT __declspec(dllexport)
#endif

#ifdef __cplusplus
extern "C" {
#endif


/* The gateway function */
extern DLL_EXPORT int cpuTone(int len, double freq, double fs, double phase, Ipp64fc *tone){
	ippInit();

    // computation
    double usedFreq; // used in the 2nd and 3rd cases
    if (freq>=0 && freq<fs){
        ippsTone_64fc(tone, len, 1.0, freq/fs, &phase, ippAlgHintAccurate);
    }
    else if (freq<0){
        usedFreq = freq;
        while (usedFreq < 0){
            usedFreq = usedFreq + fs;
        }
        ippsTone_64fc(tone, len, 1.0, usedFreq/fs, &phase, ippAlgHintAccurate);
    }
    else{
        usedFreq = freq;
        while (usedFreq >= fs){
            usedFreq = usedFreq - fs;
        }
        ippsTone_64fc(tone, len, 1.0, usedFreq/fs, &phase, ippAlgHintAccurate);
    }
    
    return 0;
	
}

#ifdef __cplusplus
}
#endif
