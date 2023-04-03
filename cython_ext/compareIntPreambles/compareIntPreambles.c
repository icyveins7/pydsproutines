// cl /O2 /LD 

#include <stdio.h>
#include <stdint.h>

#define DLL_EXPORT __declspec(dllexport)

#ifdef __cplusplus
extern "C" {
#endif

/* The gateway function */
extern DLL_EXPORT int compareIntPreambles(
    const uint8_t *preamble,
    const int preambleLength,
    const uint8_t *x,
    const int xlength,
    const int searchStart,
    const int searchEnd,
    const int m,
    uint32_t *matches // assumed to be pre-zeroed
){
    // Loop over the search
    for (int i = searchStart; i < searchEnd; i++)
    {
        /* V2 */
        for (int j = 0; j < preambleLength; j++)
        {
            if (i+j > xlength)
                return 1;
            
            // Calculate the proper rotation by adding m to the input signal
            matches[i * m + ((preamble[j] + m - x[i+j]) % m)] += 1;
        }

        /* V1 */
        // // Loop over the rotations
        // for (int rot = 0; rot < m; rot++)
        // {
        //     // and then loop over the entire preamble
        //     for (int j = 0; j < preambleLength; j++)
        //     {
        //         if (i+j > xlength)
        //             return 1; // should not go out of bounds

        //         if (preamble[j] == ((x[i+j] + rot) % m))
        //             matches[i * m + rot] += 1;
        //     }
        // }
    }
	
	return 0;
}


#ifdef __cplusplus
}
#endif