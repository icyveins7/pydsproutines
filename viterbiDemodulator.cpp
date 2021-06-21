#include <iostream>
#include <vector>
#include <stdint.h>
#include "ipp.h"
#include "ipp_ext.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
//#include <atomic>

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

		// Filter requirements (create one spec for each pulse, one buffer for each thread)
		std::vector<IppsFIRSpec_64fc*> pSpecVec;
		std::vector<Ipp8u*> bufVec_thd;
        
		// Runtime vectors (resized in methods)
		std::vector<ippe::vector<Ipp8u>> paths_index;
        ippe::vector<Ipp64f> pathmetrics;
		std::vector<ippe::vector<Ipp8u>> temppaths_index;
		ippe::vector<Ipp64f> temppathmetrics;
		
		std::vector<std::vector<Ipp64f>> branchmetrics; // use std vector for these so as to enable <algorithm> iterator uses
		std::vector<std::vector<Ipp64f>> shortbranchmetrics;

		// Workspace vectors, each symbol in the alphabet requires one, so that one thread can be assigned to each
		std::vector<ippe::vector<Ipp8u>> guess_index_thd;
		std::vector<ippe::vector<Ipp64fc>> x_sum_thd;
		std::vector<std::vector<ippe::vector<Ipp64fc>>> x_all_thd;
		std::vector<ippe::vector<Ipp64fc>> oneValArray_thd;
		std::vector<ippe::vector<Ipp64fc>> delaySrc_thd;
		std::vector<ippe::vector<Ipp64fc>> branchArray_thd;

		std::vector<std::thread> thd;
		std::vector<Ipp8u> ready; // DO NOT USE VECTOR<BOOL> FOR THESE 2
		std::vector<Ipp8u> processed;
		std::vector<std::mutex> mut;
		std::vector<std::condition_variable> cv;
		//std::atomic<int> numprocessed; // using this is too slow

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

		// Pulse filter preparation is done in ctor, before threading is selected, so just allocate for the threading anyway
		void preparePulseFilters()
		{
			// Calculate the size requirements
			int             specSize, bufSize;
			IppStatus status;
			// Get sizes of the spec structure and the work buffer
			status = ippsFIRSRGetSize(pulselen, ipp64fc, &specSize, &bufSize);
		
			// Create a spec structure for each source (i.e. each pulse)
			for (int i = 0; i < numSrc; i++) {
				IppsFIRSpec_64fc *pSpec = (IppsFIRSpec_64fc*)ippsMalloc_8u(specSize);
				ippsFIRSRInit_64fc(pulses.at(i).data(), pulselen, ippAlgDirect, pSpec); // test ippAlgFFT?
				// add to the vectors
				pSpecVec.push_back(pSpec);
			}

			// Create a buffer for each thread
			for (int i = 0; i < alphabet.size(); i++){
				Ipp8u *buf = ippsMalloc_8u(bufSize);
				// add to the vector
				bufVec_thd.push_back(buf);
			}

		}

		void freePulseFilters()
		{
			for (auto i : pSpecVec) {
				ippsFree(i);
			}
			for (auto i : bufVec_thd) {
				ippsFree(i);
			}
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
			for (int i = 0; i < paths_index.size(); i++) {
				printf("%d: ", i);
				for (int j = s; j <= n; j++) {
					printf("%.1g+%.1gi ", alphabet.at(paths_index.at(i).at(j)).re, alphabet.at(paths_index.at(i).at(j)).im);
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

		int getWorkspaceIdx(int s) {
			if (useThreading) { return s; }
			else { return 0; }
		}

		void prepareBranchMetricWorkspace(int pathlen)
		{
			// Allocate number of arrays according to threading use
			int alphabetlen = alphabet.size();
			int numThds;
			if (useThreading) { numThds = alphabet.size(); }
			else { numThds = 1; }

			guess_index_thd.resize(numThds);
			x_sum_thd.resize(numThds);
			x_all_thd.resize(numThds);
			oneValArray_thd.resize(numThds);
			delaySrc_thd.resize(numThds);
			branchArray_thd.resize(numThds);

			// Threads, mutexes, cond variables
			thd.resize(numThds);
			ready.resize(numThds);
			processed.resize(numThds);
			//mut.resize(numThds); // mutexes are not resizable in a vector due to no copy constructor
			//cv.resize(numThds); // same for condition vars
			mut = std::vector<std::mutex>(numThds); // directly re-construct it as a workaround?
			cv = std::vector<std::condition_variable>(numThds);

			for (int i = 0; i < numThds; i++) {
				// Allocate guesses
				guess_index_thd.at(i).resize(pathlen);

				// Allocate workspace (actually can move these into ctor?)
				x_sum_thd.at(i).resize(pulselen);
				x_all_thd.at(i).resize(numSrc);
				for (int j = 0; j < numSrc; j++) {
					x_all_thd.at(i).at(j).resize(pulselen);
				}
				oneValArray_thd.at(i).resize(pulselen);
				delaySrc_thd.at(i).resize(pulselen - 1);
				branchArray_thd.at(i).resize(pulselen);
			}
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
			temppaths_index.resize(alphabet.size());

			for (int i = 0; i < paths_index.size(); i++) {
				temppaths_index.at(i).resize(pathlen);
				ippsSet_8u(IPP_MAX_8U, temppaths_index.at(i).data(), temppaths_index.at(i).size());
				
				paths_index.at(i).resize(pathlen);
				ippsSet_8u(IPP_MAX_8U, paths_index.at(i).data(), paths_index.at(i).size());
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
				// Point references
				int t = getWorkspaceIdx(a);
				ippe::vector<Ipp8u> &guess_index = guess_index_thd.at(t);
				ippe::vector<Ipp64fc> &x_sum = x_sum_thd.at(t);
				std::vector<ippe::vector<Ipp64fc>> &x_all = x_all_thd.at(t);
				ippe::vector<Ipp64fc> &oneValArray = oneValArray_thd.at(t);
				ippe::vector<Ipp64fc> &delaySrc = delaySrc_thd.at(t);
				ippe::vector<Ipp64fc> &branchArray = branchArray_thd.at(t);

				// Only conduct the path metric for the set indices
				auto findresult = std::find(allowedStartSymbolIndices.begin(), allowedStartSymbolIndices.end(), (Ipp8u)a);
				if (findresult != std::end(allowedStartSymbolIndices))
				{
					printf("Calculating first symbol path metric directly for index %d\n", a);

					paths_index.at(a).at(0) = a;
					calcBranchMetricSingle(0, y, 
						paths_index.at(a), // we directly just use the paths_index
						oneValArray,
						delaySrc,
						x_all,
						bufVec_thd.at(a));

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

			// Pre-launch threads
			if (useThreading) {
				for (int t = 0; t < thd.size(); t++) {
					ready.at(t) = false;
					processed.at(t) = false;

					thd.at(t) = std::thread(&ViterbiDemodulator::calcBranchMetricsInnerPrelaunch, this, t, y, pathlen,
						std::ref(guess_index_thd.at(t)), // one workspace vector each
						std::ref(x_sum_thd.at(t)),
						std::ref(x_all_thd.at(t)),
						std::ref(oneValArray_thd.at(t)),
						std::ref(delaySrc_thd.at(t)),
						std::ref(branchArray_thd.at(t)),
						bufVec_thd.at(t));
				}
			}

			for (int n = 1; n < pathlen; n++) 
			{
				// Branches
				calcBranchMetrics(y, n, pathlen);

				// Paths
				calcPathMetrics(n);

				//// Debug
				//printBranchMetrics();
				//printPathMetrics();
				//if (n % 100 == 0) {
				//	printf("Finished %d\n", n);
				//}
			}
			
			// Close all prelaunched threads
			if (useThreading) {
				for (int t = 0; t < thd.size(); t++) {
					thd.at(t).join(); // close all threads
				}
			}

			auto t2 = std::chrono::high_resolution_clock::now();
			auto timetaken = std::chrono::duration<double>(t2 - t1).count();
			printf("Time taken for run = %f s.\n", timetaken);

			// Dump output
			dumpOutput();
		}


        void calcBranchMetrics(Ipp64fc *y, int n, int pathlen)
        {
			int t;

			//numprocessed = 0;

			//printf("\n");
			//for (int d = 0; d < processed.size(); d++) {
			//	printf("addr: processed[%d] = %p\n", d, &processed.at(d));
			//}

			// Select current symbol
			for (int p = 0; p < alphabet.size(); p++)
			{
				t = getWorkspaceIdx(p);

				if (useThreading) {
					// For pre-launched threads, we signal
					{
						std::lock_guard<std::mutex> lk(mut.at(t));
						processed.at(t) = 0;
						//printf("Set processed[%d] to %d from main, addr:%p\n", t, (int)processed.at(t), &processed.at(t));
						ready.at(t) = 1;
					}
					cv.at(t).notify_one();

					//// Here is where we spawn threads for each symbol in the alphabet
					//thd.at(t) = std::thread(&ViterbiDemodulator::calcBranchMetricsInner, this, p, y, n, pathlen,
					//	std::ref(guess_index_thd.at(t)), // one workspace vector each
					//	std::ref(x_sum_thd.at(t)),
					//	std::ref(x_all_thd.at(t)),
					//	std::ref(oneValArray_thd.at(t)),
					//	std::ref(delaySrc_thd.at(t)),
					//	std::ref(branchArray_thd.at(t)),
					//	bufVec_thd.at(t));
				}
				else {
					// Inner function for looping over preTransitions for current symbol
					calcBranchMetricsInner(p, y, n, pathlen,
						guess_index_thd.at(t), // this should be t = 0 for non-threading anyway
						x_sum_thd.at(t),
						x_all_thd.at(t),
						oneValArray_thd.at(t),
						delaySrc_thd.at(t),
						branchArray_thd.at(t),
						bufVec_thd.at(t));
				}
			} // end of loop over symbol

			// If threading, wait for them
			if (useThreading) {
				// Pre-launched threads: just wait for processed to be signalled
				for (t = 0; t < thd.size(); t++) {
					while (ready.at(t) == 1 || processed.at(t) == 0) {
					//while (numprocessed<thd.size()){
						std::unique_lock<std::mutex> lk(mut.at(t));
						//cv.at(t).wait_for(lk, std::chrono::duration<double>(0.1), [&] {printf("processed[%d]=%d\n", t, (int)processed.at(t)); return (bool)processed.at(t); });
						cv.at(t).wait_for(lk, std::chrono::duration<double>(0.1), [&] {return (bool)processed.at(t); });
						//cv.at(t).wait_for(lk, std::chrono::duration<double>(0.1), [&] {return (numprocessed==thd.size()); }); // atomics too slow
						// time out and try again because sometimes the worker thread is too fast and notifies before this call
						//printf("Timed out %d, %d\n", n, t);
					}
					// When done, reset processed (ready should have been reset inside thread)
					//processed.at(t) = false;
					//printf("Pre-launched %d complete.\n", t);
				}
				//printf("Pre-launch for %d complete\n", n);

				//for (t = 0; t < thd.size(); t++) {
				//	thd.at(t).join();
				//}
			}
        }

		/// <summary>
		/// This function wraps the standard threaded / single-threaded function but uses condition variables to signal. Note that 'n' is not an argument here.
		/// </summary>
		/// <param name="p"> Symbol index, corresponds to thread index </param>
		void calcBranchMetricsInnerPrelaunch(int p, Ipp64fc *y, int pathlen,
			ippe::vector<Ipp8u> &guess_index, // workspace vectors
			ippe::vector<Ipp64fc> &x_sum,
			std::vector<ippe::vector<Ipp64fc>> &x_all,
			ippe::vector<Ipp64fc> &oneValArray,
			ippe::vector<Ipp64fc> &delaySrc,
			ippe::vector<Ipp64fc> &branchArray,
			Ipp8u *buf)
		{
			printf("Pre-launch for %d success.\n", p);

			for (int n = 1; n < pathlen; n++) {
				// Wait til main thread is ready
				std::unique_lock<std::mutex> lk(mut.at(p));
				cv.at(p).wait(lk, [&]{return (bool)ready.at(p); });

				//printf("Pre-launched %d, n=%d ready->processing\n", p, n);

				// Process using standard function
				calcBranchMetricsInner(p, y, n, pathlen,
					guess_index,
					x_sum,
					x_all,
					oneValArray,
					delaySrc,
					branchArray,
					buf);

				// Send back to main thread
				processed.at(p) = 1;
				//if (processed.at(p) != 1) {
				//	printf("WTF IT FAILED RIGHT AFTER IT WAS SETTTTTTTTTT %d\n", p);
				//}
				ready.at(p) = 0; // so that it will not relock again
				//numprocessed++;

				lk.unlock();
				cv.at(p).notify_one();

				//printf("Pre-launched %d, n=%d notified, processedbool=%d\n", p, n, (int)processed.at(p));
				//if (processed.at(p) != 1) {
				//	printf("%d: why didn't it change in %d??\n", n, p);
				//} // BECAUSE HERE IT ALREADY NOTIFIED, THE MAIN THREAD COULD HAVE SWITCHED IT BACK ALREADY

			}
		}

		/// <param name="p"> Symbol index </param>
		void calcBranchMetricsInner(int p, Ipp64fc *y, int n, int pathlen,
									ippe::vector<Ipp8u> &guess_index, // workspace vectors
									ippe::vector<Ipp64fc> &x_sum,
									std::vector<ippe::vector<Ipp64fc>> &x_all,
									ippe::vector<Ipp64fc> &oneValArray,
									ippe::vector<Ipp64fc> &delaySrc,
									ippe::vector<Ipp64fc> &branchArray,
									Ipp8u *buf)
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

					calcBranchMetricSingle(n, y,
						guess_index,
						oneValArray,
						delaySrc,
						x_all,
						buf);

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
		/// <param name="guess_index"> Vector containing the in-order chain of symbols' alphabet indices, including the new symbol at index n </param>
		void calcBranchMetricSingle(int n, Ipp64fc *y, 
			ippe::vector<Ipp8u> &guess_index,
			ippe::vector<Ipp64fc> &oneValArray,
			ippe::vector<Ipp64fc> &delaySrc,
			std::vector<ippe::vector<Ipp64fc>> &x_all,
			Ipp8u *buf)
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
				ippsFIRSR_64fc(oneValArray.data(), x_all.at(i).data(), pulselen, pSpecVec.at(i), delaySrc.data(), NULL, buf);
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
				}
				else
				{
					ippsMinIndx_64f(branchmetrics.at(p).data(), branchmetrics.at(p).size(), &minVal, &bestPrevIdx);

					//printf("Best branch to %d is from %d, minval = %.8f\n", p, preTransitions.at(p).at(bestPrevIdx), minVal);

					ippsCopy_8u(paths_index.at(preTransitions.at(p).at(bestPrevIdx)).data(), temppaths_index.at(p).data(), temppaths_index.at(p).size());
					temppaths_index.at(p).at(n) = (Ipp8u)p;

					// Update the path metric
					temppathmetrics.at(p) = pathmetrics.at(preTransitions.at(p).at(bestPrevIdx)) + shortbranchmetrics.at(p).at(bestPrevIdx);
				}
			}

			// Write back to main vectors
			for (int i = 0; i < paths_index.size(); i++) {
				ippsCopy_8u(temppaths_index.at(i).data(), paths_index.at(i).data(), temppaths_index.at(i).size());
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

		void dumpOutput()
		{
			// Used for debugging
			printPathMetrics();

			FILE *fp = fopen("paths_index.bin", "wb");
			for (int i = 0; i < paths_index.size(); i++) {
				fwrite(paths_index.at(i).data(), sizeof(Ipp8u), paths_index.at(i).size(), fp);
			}
			fclose(fp);

			fp = fopen("pathmetrics.bin", "wb");
			fwrite(pathmetrics.data(), sizeof(Ipp64f), pathmetrics.size(), fp);
			fclose(fp);
		}

};

int main(int argc, char *argv[])
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
	if (strcmp(argv[1], "0") == 0) {
		vd.run(y.data(), y.size(), 8000);
	}

	if (strcmp(argv[1], "1") == 0) {
		vd.setUseThreading(true);
		vd.run(y.data(), y.size(), 8000);
	}

	//vd.printOmegaVectors(64000, 64010);
    
    return 0;
}