#include <iostream>
#include <vector>
#include <stdint.h>
#include "ipp_ext.h"

class ViterbiDemodulator
{
    protected:
        ippe::vector<Ipp64fc> alphabet;
        std::vector<ippe::vector<Ipp8u>> preTransitions;
        Ipp8u numSrc;
        std::vector<ippe::vector<Ipp64fc>> pulses;
        ippe::vector<Ipp64fc> omegas;
        Ipp32u up;
        
        std::vector<ippe::vector<Ipp64fc>> paths;
        ippe::vector<Ipp64fc> pathmetrics;
    
    public:
        ViterbiDemodulator(Ipp64fc *in_alphabet, uint8_t alphabetLen,
                            uint8_t *in_preTransitions, uint8_t preTransitionsLen,
                            uint8_t in_numSrc,
                            Ipp64fc *in_pulses, uint32_t pulseLen, // (numSrc * pulseLen)
                            Ipp64fc *in_omegas, // (numSrc)
                            uint32_t in_up)
                            : numSrc{in_numSrc}, up{in_up}
                            {

            // Alphabet
            alphabet.resize(alphabetLen);
            ippsCopy_64fc(in_alphabet, alphabet.data(), alphabet.size());
            
            // Pretransitions
            preTransitions.resize(alphabetLen);
            for (int i = 0; i < alphabetLen; i++){
                for (int j = 0; j < preTransitionsLen; j++){
                    preTransitions.at(i).push_back(in_preTransitions[i*preTransitionsLen+j]);
                }
            }
            
            // // Pulses
            // pulses.resize(numSrc);
            // for (int i = 0; i < numSrc; i++){
                // pulses.at(i).resize(pulseLen);
                // ippsCopy_64fc(&in_pulses[i*pulseLen], pulses.at(i).data(), pulseLen);
            // }
            // // Omegas
            // omegas.resize(numSrc);
            // for (int i = 0; i < numSrc; i++){
                // omegas.push_back(in_omegas[i]);
            // }
            
            printf("ViterbiDemodulator initialized.\n");
        
        }
                                
        ~ViterbiDemodulator()
        {
        }

        void printAlphabet()
        {
            printf("Alphabet:\n");
            for (int i = 0; i < alphabet.size(); i++){
                printf("%g + %gi\n", alphabet.at(i).re, alphabet.at(i).im);
            }
            printf("=====\n");
        }
        
        void printValidTransitions()
        {
            printf("Valid transitions:\n");
            for (int i = 0; i < preTransitions.size(); i++){
                for (int j = 0; j < preTransitions.at(i).size(); j++){
                    printf("%d->%d\n", preTransitions.at(i).at(j), i);
                }
            }
            printf("=====\n");
        }
        
        void branchMetric()
        {
            
            
        }
        void pathMetric()
        {
            
        }

};

int main()
{
    ippe::vector<Ipp64fc> alphabet(4);
    alphabet.at(0) = {1.0, 0.0};
    alphabet.at(1) = {0.0, 1.0};
    alphabet.at(2) = {-1.0, 0.0};
    alphabet.at(3) = {0.0, -1.0};
    
    ippe::vector<Ipp8u> pretransitions;
    pretransitions.push_back(1);
    pretransitions.push_back(3);
    pretransitions.push_back(0);
    pretransitions.push_back(2);
    pretransitions.push_back(1);
    pretransitions.push_back(3);
    pretransitions.push_back(0);
    pretransitions.push_back(2);
    
    ViterbiDemodulator vd(alphabet.data(), alphabet.size(),
                            pretransitions.data(), 2,
                            5,
                            nullptr, 200, // (numSrc * pulseLen)
                            nullptr, // (numSrc)
                            8);
                            
    vd.printAlphabet();
    vd.printValidTransitions();
    
    
    return 0;
}