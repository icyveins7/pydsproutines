#pragma once
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

public:
	ViterbiDemodulator(Ipp64fc* in_alphabet, uint8_t alphabetLen,
		uint8_t* in_preTransitions, uint8_t preTransitionsLen,
		uint8_t in_numSrc,
		Ipp64fc* in_pulses, int in_pulselen, // (numSrc * pulseLen)
		Ipp64f* in_omegas, // (numSrc)
		uint32_t in_up);

	~ViterbiDemodulator();

	void preparePulseFilters();
	void freePulseFilters();

	// Printers
	std::string printAlphabet();
	std::string printValidTransitions();
	std::string printOmega();
	void printPathMetrics();
	void printPaths(int n, int s = 0);
	void printBranchMetrics();
	std::string printOmegaVectors(int s, int e);
	std::string printPulses(int s, int e);

	// Get/Set
	int getWorkspaceIdx(int s);
	void setAllowedStartSymbolIndices(std::vector<Ipp8u> newAllowedIndices = {});
	void setUseThreading(bool in);

	// Sub-runtime methods
	void prepareBranchMetricWorkspace(int pathlen);
	void calcBranchMetrics(Ipp64fc* y, int n, int pathlen);
	void calcBranchMetricsInnerPrelaunch(int p, Ipp64fc* y, int pathlen,
		ippe::vector<Ipp8u>& guess_index, // workspace vectors
		ippe::vector<Ipp64fc>& x_sum,
		std::vector<ippe::vector<Ipp64fc>>& x_all,
		ippe::vector<Ipp64fc>& oneValArray,
		ippe::vector<Ipp64fc>& delaySrc,
		ippe::vector<Ipp64fc>& branchArray,
		Ipp8u* buf);
	void calcBranchMetricsInner(int p, Ipp64fc* y, int n, int pathlen,
		ippe::vector<Ipp8u>& guess_index, // workspace vectors
		ippe::vector<Ipp64fc>& x_sum,
		std::vector<ippe::vector<Ipp64fc>>& x_all,
		ippe::vector<Ipp64fc>& oneValArray,
		ippe::vector<Ipp64fc>& delaySrc,
		ippe::vector<Ipp64fc>& branchArray,
		Ipp8u* buf);
	void calcBranchMetricSingle(int n, Ipp64fc* y,
		ippe::vector<Ipp8u>& guess_index,
		ippe::vector<Ipp64fc>& oneValArray,
		ippe::vector<Ipp64fc>& delaySrc,
		std::vector<ippe::vector<Ipp64fc>>& x_all,
		Ipp8u* buf);
	void calcPathMetrics(int n);
	void prepareOmegaVectors(int length);

	// Main run-time method
	int run(Ipp64fc* y, int ylength, int pathlen);

	// Debugging
	void dumpOutput();
};