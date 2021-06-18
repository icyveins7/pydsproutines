#include <iostream>
#include <vector>
#include <stdint.h>
#include "ipp.h"
#include "ipp_ext.h"
#include <cmath>
#include <chrono>
#include <algorithm>

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

		// Settings (with defaults)
		std::vector<Ipp8u> allowedStartSymbolIndices = { 0 };
		bool useThreading = false;

		// Useful constants
		Ipp64fc inf64fc = { std::numeric_limits<double>::infinity() , std::numeric_limits<double>::infinity() };

		// Omega tone vectors
		std::vector<ippe::vector<Ipp64fc>> omegavectors;

		// Filter requirements (create one for each pulse)
		std::vector<IppsFIRSpec_64fc*> pSpecVec;
		std::vector<Ipp8u*> bufVec;
        
		// Runtime vectors (resized in methods)
		std::vector<ippe::vector<Ipp8u>> paths_index;
        std::vector<ippe::vector<Ipp64fc>> paths;
        ippe::vector<Ipp64f> pathmetrics;
		std::vector<ippe::vector<Ipp8u>> temppaths_index;
		std::vector<ippe::vector<Ipp64fc>> temppaths;
		ippe::vector<Ipp64f> temppathmetrics;
		
		std::vector<std::vector<Ipp64f>> branchmetrics; // use std vector for these so as to enable <algorithm> iterator uses
		std::vector<std::vector<Ipp64f>> shortbranchmetrics;

		// Workspace vectors
		ippe::vector<Ipp8u> guess_index;
		ippe::vector<Ipp64fc> guess;
		//ippe::vector<Ipp64fc> upguess;

		ippe::vector<Ipp64fc> x_sum;
		std::vector<ippe::vector<Ipp64fc>> x_all;
		ippe::vector<Ipp64fc> oneValArray;
		ippe::vector<Ipp64fc> delaySrc;
		ippe::vector<Ipp64fc> branchArray;
    
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
                 omegas.at(i) = in_omegas[i];
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

		void setAllowedStartSymbolIndices(std::vector<Ipp8u> newAllowedIndices = {})
		{
			if (newAllowedIndices.size() > 0) {
				// Assign the new indices
				allowedStartSymbolIndices = newAllowedIndices;
				printf("Start symbol indices have been changed.\n");
			}
			printf("Start symbol indices are now:\n");
			for (auto i : allowedStartSymbolIndices) {
				printf("%hhu \n", i);
			}
		}

		void setUseThreading(bool in)
		{
			useThreading = in;
		}

		void printPathMetrics()
		{
			printf("Path metrics:\n");
			for (int i = 0; i < pathmetrics.size(); i++)
			{
				printf("%.8f\n", pathmetrics.at(i));
			}
			printf("\n");
		}

		void printPaths(int n, int s=0)
		{
			printf("Paths: \n");
			for (int i = 0; i < paths.size(); i++) {
				printf("%d: ", i);
				for (int j = s; j <= n; j++) {
					printf("%.1g+%.1gi ", paths.at(i).at(j).re, paths.at(i).at(j).im);
				}
				printf("\n");
			}
		}

		void printBranchMetrics()
		{
			printf("Branch metrics:\n");
			for (int i = 0; i < branchmetrics.size(); i++) {
				printf("%d: ", i);
				for (auto j : branchmetrics.at(i)) {
					printf("%.8f ", j);
				}
				printf("\n");
			}
			printf("Short branch metrics:\n");
			for (int i = 0; i < shortbranchmetrics.size(); i++) {
				printf("%d: ", i);
				for (auto j : shortbranchmetrics.at(i)) {
					printf("%.8f ", j);
				}
				printf("\n");
			}
			printf("\n");
		}

		void printOmegaVectors(int s, int e)
		{
			printf("Omega vectors:\n");
			for (int i = 0; i < omegavectors.size(); i++) {
				for (int j = s; j < e; j++) {
					printf("%d,%d : %.8f + %.8fi\n", i, j, omegavectors.at(i).at(j).re, omegavectors.at(i).at(j).im);
				}
			}
			printf("\n");
		}

        
		// Runtime
		void run(Ipp64fc *y, int ylength, int pathlen)
		{
			// Time the run
			auto t1 = std::chrono::high_resolution_clock::now();

			// Pregenerate omega vectors
			prepareOmegaVectors(ylength);

			// Allocate paths
			paths_index.resize(alphabet.size());
			paths.resize(alphabet.size());
			temppaths_index.resize(alphabet.size());
			temppaths.resize(alphabet.size());

			for (int i = 0; i < paths.size(); i++) {
				temppaths_index.at(i).resize(pathlen);
				ippsSet_8u(IPP_MAX_8U, temppaths_index.at(i).data(), temppaths_index.at(i).size());
				
				paths_index.at(i).resize(pathlen);
				ippsSet_8u(IPP_MAX_8U, paths_index.at(i).data(), paths_index.at(i).size());

				paths.at(i).resize(pathlen);
				ippsZero_64fc(paths.at(i).data(), paths.at(i).size());

				temppaths.at(i).resize(pathlen);
				ippsZero_64fc(temppaths.at(i).data(), temppaths.at(i).size());
			}

			// Allocate pathmetrics
			pathmetrics.resize(alphabet.size());
			temppathmetrics.resize(alphabet.size());
			ippsSet_64f(std::numeric_limits<double>::infinity(), pathmetrics.data(), pathmetrics.size());

			// Allocate branchmetrics (the resize shouldn't do anything beyond the first call, hopefully not a performance hit from checking)
			branchmetrics.resize(preTransitions.size());
			for (int i = 0; i < branchmetrics.size(); i++) {
				branchmetrics.at(i).resize(preTransitions.at(i).size());
			}

			shortbranchmetrics.resize(preTransitions.size());
			for (int i = 0; i < shortbranchmetrics.size(); i++) {
				shortbranchmetrics.at(i).resize(preTransitions.at(i).size());
			}

			// Workspace for branches
			prepareBranchMetricWorkspace(pathlen);
			
			// Construct the path metric for the first symbol
			for (int a = 0; a < alphabet.size(); a++)
			{
				// Only conduct the path metric for the set indices
				auto findresult = std::find(allowedStartSymbolIndices.begin(), allowedStartSymbolIndices.end(), (Ipp8u)a);
				if (findresult != std::end(allowedStartSymbolIndices))
				{
					printf("Calculating first symbol path metric directly for index %d\n", a);

					paths_index.at(a).at(0) = a;
					calcBranchMetricSingle(0, y, paths_index.at(a));

					//paths.at(a).at(0) = alphabet.at(a);
					//calcBranchMetricSingle(0, y, paths.at(a));

					// Sum all the sources, along with the multiply of the omegavector
					ippsZero_64fc(x_sum.data(), x_sum.size());
					for (int i = 0; i < numSrc; i++) {
						ippsAddProduct_64fc(x_all.at(i).data(), &omegavectors.at(i).at(0*up), x_sum.data(), pulselen);
					}

					// Calculate the branchArray and the norms from it
					ippsSub_64fc(x_sum.data(), &y[0*up], branchArray.data(), x_sum.size());
					// But for first one save directly to pathmetric
					ippsNorm_L2_64fc64f(branchArray.data(), up, &pathmetrics.at(a));
					// Squaring done separately
					pathmetrics.at(a) = pathmetrics.at(a) * pathmetrics.at(a);
				}
				else {
					printf("Skipping first symbol path metric for index %d\n", a);
				}
			}
			printPathMetrics();

			// Loop over the rest of the symbols
			printf("Beginning loop over symbols..\n");
			for (int n = 1; n < pathlen; n++) 
			{
				// Branches
				calcBranchMetrics(y, n, pathlen);

				// Paths
				calcPathMetrics(n);

				//// Debug
				//printBranchMetrics();
				//printPathMetrics();
			}

			auto t2 = std::chrono::high_resolution_clock::now();
			auto timetaken = std::chrono::duration<double>(t2 - t1).count();
			printf("Time taken for run = %f s.\n", timetaken);

			// Dump output
			dumpOutput();
		}


        void calcBranchMetrics(Ipp64fc *y, int n, int pathlen)
        {
			// Select current symbol
			for (int p = 0; p < alphabet.size(); p++)
			{
				// Inner function for looping over preTransitions for current symbol
				calcBranchMetricsInner(p, y, n, pathlen);
			} // end of loop over symbol
        }

		void calcBranchMetricsInner(int p, Ipp64fc *y, int n, int pathlen)
		{
			// Select a pre-transition path
			for (int t = 0; t < preTransitions.at(p).size(); t++)
			{
				if (pathmetrics.at(preTransitions.at(p).at(t)) == std::numeric_limits<double>::infinity())
				{
					branchmetrics.at(p).at(t) = std::numeric_limits<double>::infinity();
					shortbranchmetrics.at(p).at(t) = std::numeric_limits<double>::infinity();
				}
				else // Main process
				{
					//printf("%4d: Pre-transition %d->%d\n", n, preTransitions.at(p).at(t), p);

					// Copy values to guess index instead
					int num2copy = pulselen / up + 1; // this is the maximum number of symbols that will be required, no need to copy the entire path
					num2copy = IPP_MIN(num2copy, n);
					ippsCopy_8u(&paths_index.at(preTransitions.at(p).at(t)).at(n - num2copy), &guess_index.at(n - num2copy), num2copy); // this has very insignificant change, probably only see it at very long pathlen (in fact, removing this line entirely has almost no impact on the timing)
					//ippsCopy_8u(paths_index.at(preTransitions.at(p).at(t)).data(), guess_index.data(), pathlen);
					guess_index.at(n) = (Ipp8u)p;

					//// Copy values over to guess
					//ippsCopy_64fc(paths.at(preTransitions.at(p).at(t)).data(), guess.data(), pathlen);
					//guess.at(n) = alphabet.at(p); // Set new value for this index

					// Don't need this in C++!
					//int uplen = upguess.size();
					//int phase = 0;
					//ippsSampleUp_64fc(guess, pathlen, upguess, &uplen, up, &phase);

					calcBranchMetricSingle(n, y, guess_index);
					//calcBranchMetricSingle(n, y, guess);

					// Sum all the sources, along with the multiply of the omegavector
					ippsZero_64fc(x_sum.data(), x_sum.size());
					for (int i = 0; i < numSrc; i++) {
						ippsAddProduct_64fc(x_all.at(i).data(), &omegavectors.at(i).at(n*up), x_sum.data(), pulselen);
					}

					// Calculate the branchArray and the norms from it
					ippsSub_64fc(x_sum.data(), &y[n*up], branchArray.data(), x_sum.size());
					ippsNorm_L2_64fc64f(branchArray.data(), branchArray.size(), &branchmetrics.at(p).at(t));
					ippsNorm_L2_64fc64f(branchArray.data(), up, &shortbranchmetrics.at(p).at(t));

					// Squaring done separately
					branchmetrics.at(p).at(t) = branchmetrics.at(p).at(t) * branchmetrics.at(p).at(t);
					shortbranchmetrics.at(p).at(t) = shortbranchmetrics.at(p).at(t) * shortbranchmetrics.at(p).at(t);


				}
			} // end of loop over pretransition
		}

		/// <summary>
		/// Calculates the branch for symbol at path index n
		/// </summary>
		/// <param name="guess"> Vector containing the in-order chain of symbols, including the new symbol at index n </param>
		//void calcBranchMetricSingle(int n, Ipp64fc *y, ippe::vector<Ipp64fc> &guess)
		void calcBranchMetricSingle(int n, Ipp64fc *y, ippe::vector<Ipp8u> &guess_index)
		{
			int guessIdx;
			// Get tracking index
			int s = IPP_MAX(n * up - pulselen, 0);
			// Loop over the sources
			for (int i = 0; i < numSrc; i++) {
				// Set the one value array
				ippsZero_64fc(oneValArray.data(), oneValArray.size());
				//oneValArray.at(0) = guess.at(n);
				oneValArray.at(0) = alphabet.at(guess_index.at(n)); // convert the index to the complex valued symbol

				// Construct the delay line backwards
				ippsZero_64fc(delaySrc.data(), delaySrc.size());
				for (int j = up; j < delaySrc.size(); j = j + up) {
					guessIdx = n - j / up;
					if (guessIdx >= 0) {
						//delaySrc.at(delaySrc.size() - j) = guess.at(guessIdx);
						delaySrc.at(delaySrc.size() - j) = alphabet.at(guess_index.at(guessIdx)); // convert the index to the complex valued symbol
					}
				}

				// Filter
				ippsFIRSR_64fc(oneValArray.data(), x_all.at(i).data(), pulselen, pSpecVec.at(i), delaySrc.data(), NULL, bufVec.at(i));
			}

		}


        void calcPathMetrics(int n)
        {
			int bestPrevIdx;
			Ipp64f minVal;

			for (int p = 0; p < branchmetrics.size(); p++)
			{
				if (std::all_of(branchmetrics.at(p).begin(), branchmetrics.at(p).end(), [](Ipp64f i) {return isinf(i); }))
				{
					temppathmetrics.at(p) = std::numeric_limits<double>::infinity();

					ippsCopy_8u(paths_index.at(p).data(), temppaths_index.at(p).data(), paths_index.at(p).size());

					//ippsCopy_64fc(paths.at(p).data(), temppaths.at(p).data(), paths.at(p).size()); // vector assignment doesn't work with Ipp types
				}
				else
				{
					ippsMinIndx_64f(branchmetrics.at(p).data(), branchmetrics.at(p).size(), &minVal, &bestPrevIdx);

					//printf("Best branch to %d is from %d, minval = %.8f\n", p, preTransitions.at(p).at(bestPrevIdx), minVal);

					ippsCopy_8u(paths_index.at(preTransitions.at(p).at(bestPrevIdx)).data(), temppaths_index.at(p).data(), temppaths_index.at(p).size());
					temppaths_index.at(p).at(n) = (Ipp8u)p;

					/*ippsCopy_64fc(paths.at(preTransitions.at(p).at(bestPrevIdx)).data(), temppaths.at(p).data(), temppaths.at(p).size());
					temppaths.at(p).at(n) = alphabet.at(p);*/

					// Update the path metric
					temppathmetrics.at(p) = pathmetrics.at(preTransitions.at(p).at(bestPrevIdx)) + shortbranchmetrics.at(p).at(bestPrevIdx);
				}
			}

			// Write back to main vectors
			for (int i = 0; i < paths.size(); i++) {
				ippsCopy_8u(temppaths_index.at(i).data(), paths_index.at(i).data(), temppaths_index.at(i).size());

				//ippsCopy_64fc(temppaths.at(i).data(), paths.at(i).data(), temppaths.at(i).size());
			}
			ippsCopy_64f(temppathmetrics.data(), pathmetrics.data(), temppathmetrics.size());
        }

		void prepareOmegaVectors(int length)
		{
			Ipp64f phase;
			Ipp64f rFreq;
			IppStatus status;

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
				printf("Omega %g converted to rFreq %g\n", omegas.at(i), rFreq);
				status = ippsTone_64fc(omegavectors.at(i).data(), omegavectors.at(i).size(), 1.0, rFreq, &phase, ippAlgHintAccurate);
			}
			if (status == ippStsNoErr) {
				printf("Constructed omega vectors.\n");
			}
			else {
				printf("Error in constructing omega!\n");
			}
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

		void prepareBranchMetricWorkspace(int pathlen)
		{
			// Allocate guesses
			guess_index.resize(pathlen);
			guess.resize(pathlen);
			//upguess.resize(pathlen * up);

			// Allocate workspace (actually can move these into ctor?)
			x_sum.resize(pulselen);
			x_all.resize(numSrc);
			for (int i = 0; i < numSrc; i++) {
				x_all.at(i).resize(pulselen);
			}
			oneValArray.resize(pulselen);
			delaySrc.resize(pulselen - 1);
			branchArray.resize(pulselen);
		}

		void dumpOutput()
		{
			// Used for debugging
			printPathMetrics();

			FILE *fp = fopen("paths.bin", "wb");
			for (int i = 0; i < paths.size(); i++) {
				fwrite(paths.at(i).data(), sizeof(Ipp64fc), paths.at(i).size(), fp);
			}
			fclose(fp);

			fp = fopen("pathmetrics.bin", "wb");
			fwrite(pathmetrics.data(), sizeof(Ipp64f), pathmetrics.size(), fp);
			fclose(fp);
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
	vd.setAllowedStartSymbolIndices();

	// Test run
	vd.run(y.data(), y.size(), 8000);

	//vd.printOmegaVectors(64000, 64010);
    
    return 0;
}