#include "viterbiDemodulator.h"

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
	vd.printOmega();
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