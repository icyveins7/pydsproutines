#include <iostream>
#include <vector>
#include <stdint.h>
#include "ipp.h"
#include "ipp_ext.h"
#include <math.h>
#include <chrono>

class ViterbiDemodulator
{
    protected:
		// Constructor requirements
        ippe::vector<Ipp64fc> alphabet;
        std::vector<ippe::vector<Ipp8u>> preTransitions;
        Ipp8u numSrc;
        std::vector<ippe::vector<Ipp64fc>> pulses;
		int pulselen;
        ippe::vector<Ipp64f> omegas;
        Ipp32u up;

		// Omega tone vectors
		std::vector<ippe::vector<Ipp64fc>> omegavectors;

		// Filter requirements (create one for each pulse)
		std::vector<IppsFIRSpec_64fc*> pSpecVec;
		std::vector<Ipp8u*> bufVec;
        
		// Runtime vectors (resized in methods)
        std::vector<ippe::vector<Ipp64fc>> paths;
        ippe::vector<Ipp64f> pathmetrics;
		
		std::vector<ippe::vector<Ipp64f>> branchmetrics;
		std::vector<ippe::vector<Ipp64f>> shortbranchmetrics;
    
    public:
        ViterbiDemodulator(Ipp64fc *in_alphabet, uint8_t alphabetLen,
                            uint8_t *in_preTransitions, uint8_t preTransitionsLen,
                            uint8_t in_numSrc,
                            Ipp64fc *in_pulses, int in_pulselen, // (numSrc * pulseLen)
                            Ipp64f *in_omegas, // (numSrc)
                            uint32_t in_up)
			: numSrc{ in_numSrc }, up{ in_up }, pulselen{ in_pulselen }
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
            
             // Pulses
             pulses.resize(numSrc);
             for (int i = 0; i < numSrc; i++){
                 pulses.at(i).resize(pulselen);
                 ippsCopy_64fc(&in_pulses[i*pulselen], pulses.at(i).data(), pulselen);
             }

             // Omegas
             omegas.resize(numSrc);
             for (int i = 0; i < numSrc; i++){
                 omegas.push_back(in_omegas[i]);
             }

			 // Compute filters based on pulses
			 preparePulseFilters();
            
            printf("ViterbiDemodulator initialized.\n");
        
        }
                                
        ~ViterbiDemodulator()
        {
			freePulseFilters();
        }

        void printAlphabet()
        {
            printf("Alphabet:\n");
            for (int i = 0; i < alphabet.size(); i++){
                printf("%2g + %2gi\n", alphabet.at(i).re, alphabet.at(i).im);
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
        
		// Runtime
		void run(Ipp64fc *y, int pathlen)
		{
			// Pregenerate omega vectors
			prepareOmegaVectors();

			// Allocate paths/branchs and their metrics
			std::vector<ippe::vector<Ipp64fc>> paths;
			paths.resize(alphabet.size());
			for (int i = 0; i < paths.size(); i++) {
				paths.at(i).resize(pathlen);
			}

			ippe::vector<Ipp64f> pathmetrics;
			pathmetrics.resize(alphabet.size());
			ippsSet_64fc(INFINITY, pathmetrics.data(), pathmetrics.size());

			// Construct the path metric for the first symbol


		}


        void calcBranchMetrics(Ipp64fc *y, int n)
        {
			// probably want to set this as member variable at some point
			int pathlen = paths.size() / numSrc;

			// Allocate branchmetrics (the resize shouldn't do anything beyond the first call, hopefully not a performance hit from checking)
			branchmetrics.resize(preTransitions.size());
			for (int i = 0; i < branchmetrics.size(); i++) {
				branchmetrics.at(i).resize(preTransitions.at(i).size());
			}

			shortbranchmetrics.resize(preTransitions.size());
			for (int i = 0; i < shortbranchmetrics.size(); i++) {
				shortbranchmetrics.at(i).resize(preTransitions.at(i).size());
			}


			// Allocate guesses
			ippe::vector<Ipp64fc> guess(pathlen);
			ippe::vector<Ipp64fc> upguess(pathlen * up);

			// Allocate workspace
			ippe::vector<Ipp64fc> x_sum(pulselen);
			std::vector<ippe::vector<Ipp64fc>> x_all;
			x_all.resize(numSrc);
			for (int i = 0; i < numSrc; i++) {
				x_all.at(i).resize(pulselen);
			}
			int s;
			ippe::vector<Ipp64fc> oneValArray(pulselen);
			ippe::vector<Ipp64fc> delaySrc(pulselen - 1);
			int guessIdx;
			ippe::vector<Ipp64fc> branchArray(pulselen);


			// Select current symbol
			for (int p = 0; p < alphabet.size(); p++)
			{
				// Select a pre-transition path
				for (int t = 0; t < preTransitions.at(p).size(); t++) 
				{
					if (pathmetrics.at(preTransitions.at(p).at(t)) == INFINITY) 
					{
						branchmetrics.at(p).at(t) = INFINITY;
						shortbranchmetrics.at(p).at(t) = INFINITY;
					}
					else // Main process
					{
						// Copy values over to guess
						ippsCopy_64fc(paths.at(p).data(), guess.data(), pathlen);
						guess.at(n) = alphabet.at(p); // Set new value for this index

						// Don't need this in C++!
						//int uplen = upguess.size();
						//int phase = 0;
						//ippsSampleUp_64fc(guess, pathlen, upguess, &uplen, up, &phase);

						// Get tracking index
						s = IPP_MAX(n * up - pulselen, 0);
						// Loop over the sources
						for (int i = 0; i < numSrc; i++) {
							// Set the one value array
							ippsZero_64fc(oneValArray.data(), oneValArray.size());
							oneValArray.at(0) = guess.at(n);

							// Construct the delay line backwards
							ippsZero_64fc(delaySrc.data(), delaySrc.size());
							for (int j = up; j < delaySrc.size(); j=j+up) {
								guessIdx = n - j / up;
								if (guessIdx >= 0) {
									delaySrc.at(delaySrc.size() - j) = guess.at(guessIdx);
								}
							}
							
							// Filter
							ippsFIRSR_64fc(oneValArray.data(), x_all.at(i).data(), pulselen, pSpecVec.at(i), delaySrc.data(), NULL, bufVec.at(i));

							//// Multiply by omegavector // we get to do this below in the accumulation step with 1 instruction
							//ippsMul_64fc_I(&omegavectors.at(i).at(n*up), x_all.at(i).data(), pulselen);
							
						}

						// Sum all the sources, along with the multiply of the omegavector
						ippsZero_64fc(x_sum.data(), x_sum.size());
						for (int i = 0; i < numSrc; i++) {
							ippsAddProduct_64fc(x_all.at(i).data(), &omegavectors.at(i).at(n*up), x_sum.data(), pulselen, 0);
						}

						// Calculate the branchArray and the norms from it
						ippsSub_64fc(x_sum.data(), &y[n*up], branchArray.data(), x_sum.size());
						ippsNorm_L2_64fc64f(branchArray.data(), branchArray.size(), &branchmetrics.at(p).at(t));
						ippsNorm_L2_64fc64f(branchArray.data(), up, &shortbranchmetrics.at(p).at(t));
					}

				}
			}

            
        }
        void pathMetric()
        {
            
        }

		void prepareOmegaVectors(int length)
		{
			Ipp64f phase;
			Ipp64f rFreq;

			omegavectors.resize(numSrc);
			for (int i = 0; i < numSrc; i++) {
				omegavectors.at(i).resize(length);

				phase = 0;
				rFreq = -omegas.at(i) / IPP_2PI;
				while (rFreq < 0) {
					rFreq = rFreq + 1.0; // correct it to within [0,1) from -ve
				}
				while (rFreq >= 1.0) {
					rFreq = rFreq - 1.0; // correct it to within [0,1) from +ve above 1
				}
				ippsTone_64fc(omegavectors.at(i).data(), omegavectors.at(i).size(), 1.0, rFreq, &phase, ippAlgHintAccurate);
			}
			printf("Constructed omega vectors.");
		}

		void preparePulseFilters()
		{
			for (int i = 0; i < numSrc; i++) {
				IppsFIRSpec_64fc *pSpec;
				Ipp8u           *buf;
				int             specSize, bufSize;
				IppStatus status;
				//get sizes of the spec structure and the work buffer
				status = ippsFIRSRGetSize(pulselen, ipp64fc, &specSize, &bufSize);
				pSpec = (IppsFIRSpec_64fc*)ippsMalloc_8u(specSize);
				buf = ippsMalloc_8u(bufSize);
				//initialize the spec structure
				ippsFIRSRInit_64fc(pulses.at(i).data(), pulselen, ippAlgDirect, pSpec);
				// add to the vectors
				pSpecVec.push_back(pSpec);
				bufVec.push_back(buf);
			}
		}

		void freePulseFilters()
		{
			for (auto i : pSpecVec) {
				ippsFree(i);
			}
			for (auto i : bufVec) {
				ippsFree(i);
			}
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
    
	// load the data
	int numSrc = 4;
	int pulselen = 200;
	ippe::vector<Ipp64fc> y(64193);
	ippe::vector<Ipp64fc> pulses(numSrc * pulselen);
	ippe::vector<Ipp64f> omegas(numSrc);

	FILE *fp;
	size_t numRead;
	fp = fopen("y.bin", "rb");
	numRead = fread(y.data(), sizeof(Ipp64fc), 64193, fp);
	fclose(fp);
	printf("Read %zd values from y.bin\n", numRead);

	fp = fopen("omega.bin", "rb");
	numRead = fread(omegas.data(), sizeof(Ipp64f), numSrc, fp);
	fclose(fp);
	printf("Read %zd values from omega.bin\n", numRead);

	fp = fopen("pulses.bin", "rb");
	numRead = fread(pulses.data(), sizeof(Ipp64fc), numSrc * pulselen, fp);
	fclose(fp);
	printf("Read %zd values from pulses.bin\n", numRead);

	// check reads
	for (int i = 0; i < 5; i++) {
		printf("y[%d] = %f + %fi\n", i, y.at(i).re, y.at(i).im);
	}
	for (int i = 0; i < numSrc; i++) {
		printf("omegas[%d] = %g\n", i, omegas.at(i));
	}

	for (int i = 0; i < numSrc; i++) {
		for (int j = 0; j < pulselen; j++) {
			if (pulses.at(i*pulselen + j).re != 0) { // it should be purely real
				printf("pulses[%d,%d] = %f + %fi\n", i, j, pulses.at(i*pulselen + j).re, pulses.at(i*pulselen + j).im);
			}
		}
	}

    ViterbiDemodulator vd(alphabet.data(), alphabet.size(),
                            pretransitions.data(), 2,
                            numSrc,
                            pulses.data(), pulselen, // (numSrc * pulseLen)
                            omegas.data(), // (numSrc)
                            8);
                            
    vd.printAlphabet();
    vd.printValidTransitions();
    
    
    return 0;
}