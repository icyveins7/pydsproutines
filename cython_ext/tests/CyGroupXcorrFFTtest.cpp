// cl CyGroupXcorrFFTtest.cpp /EHsc /I.. ippcore.lib ipps.lib
// g++ CyGroupXcorrFFTtest.cpp ../GroupXcorrFFT.cpp -I../ -lippcore -lipps -lpthread -o CyGroupXcorrFFTtest

#include <iostream>
#include <vector>
#include "ipp_ext.h"
#include "ipp.h"
#include "GroupXcorrFFT.h"

int main(int argc, char *argv[])
{
	int numGroups, groupLength, shiftsLength, secLength, fs, NUM_THREADS;

	if (argc != 12)
	{
		printf(
			"Call with sizes and filenames, e.g. \n"
			".exe ygroups.bin offsets.bin shifts.bin secfilt.bin out.bin numGroups groupLength shiftsLength secLength fs NUM_THREADS\n");
		return 1;
	}
	else
	{
		numGroups = atoi(argv[6]);
		groupLength = atoi(argv[7]);
		shiftsLength = atoi(argv[8]);
		secLength = atoi(argv[9]);
		fs = atoi(argv[10]);
		NUM_THREADS = atoi(argv[11]);
	}

	// load files
	FILE* fp; size_t cnt;

	fp = fopen(argv[1], "rb");
	ippe::vector<Ipp32fc> ygroups(numGroups * groupLength);
	cnt = fread(ygroups.data(), sizeof(Ipp32fc), numGroups * groupLength, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%f, %f\n", ygroups.back().re, ygroups.back().im);

	fp = fopen(argv[2], "rb");
	ippe::vector<Ipp32s> offsets(numGroups);
	cnt = fread(offsets.data(), sizeof(Ipp32s), numGroups, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%d\n", offsets.back());

	fp = fopen(argv[3], "rb");
	ippe::vector<Ipp32s> shifts(shiftsLength);
	cnt = fread(shifts.data(), sizeof(Ipp32s), shiftsLength, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%d\n", shifts.back());

	fp = fopen(argv[4], "rb");
	ippe::vector<Ipp32fc> secfilt(secLength);
	cnt = fread(secfilt.data(), sizeof(Ipp32fc), secLength, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%f, %f\n", secfilt.back().re, secfilt.back().im);

	fp = fopen(argv[5], "rb");
	ippe::vector<Ipp32f> out(shiftsLength * groupLength);
	cnt = fread(out.data(), sizeof(Ipp32f), shiftsLength * groupLength, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%f\n", out.back());

	


	// instantiate
	GroupXcorrFFT gxcfft(
		ygroups.data(),
		numGroups, groupLength,
		offsets.data(),
		fs
	); // fftlen assumed to be same as grouplength

	// run
	ippe::vector<Ipp32f> cyout(out.size());
	gxcfft.xcorr(
		secfilt.data(), secfilt.size(),
		shifts.data(), shifts.size(),
		cyout.data(), NUM_THREADS
	);

	// compare?

	// dump?



	return 0;
}