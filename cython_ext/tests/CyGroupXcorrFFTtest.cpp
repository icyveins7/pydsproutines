// cl CyGroupXcorrFFTtest.cpp /EHsc /I.. ippcore.lib ipps.lib

#include <iostream>
#include <vector>
#include "ipp_ext.h"
#include "ipp.h"

int main()
{
	// load files
	FILE* fp; size_t cnt;

	fp = fopen("ygroups.bin", "rb");
	ippe::vector<Ipp32fc> ygroups(10 * 6860);
	cnt = fread(ygroups.data(), sizeof(Ipp32fc), 10 * 6860, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%f, %f\n", ygroups.back().re, ygroups.back().im);

	fp = fopen("offsets.bin", "rb");
	ippe::vector<Ipp32s> offsets(10);
	cnt = fread(offsets.data(), sizeof(Ipp32s), 10, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%d\n", offsets.back());

	fp = fopen("shifts.bin", "rb");
	ippe::vector<Ipp32s> shifts(16384);
	cnt = fread(shifts.data(), sizeof(Ipp32s), 16384, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%d\n", shifts.back());

	fp = fopen("secfilt.bin", "rb");
	ippe::vector<Ipp32fc> secfilt(150000000);
	cnt = fread(secfilt.data(), sizeof(Ipp32fc), 150000000, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%f, %f\n", secfilt.back().re, secfilt.back().im);

	fp = fopen("out.bin", "rb");
	ippe::vector<Ipp32f> out(16384 * 6860);
	cnt = fread(out.data(), sizeof(Ipp32f), 16384 * 6860, fp);
	printf("Read %zd elements\n", cnt);
	fclose(fp);
	printf("%f\n", out.back());






	return 0;
}