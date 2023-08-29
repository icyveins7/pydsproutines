#include "GroupXcorrCZT.h"
#include <iostream>

int main(int argc, char *argv[])
{
    // Create some data
    ippe::vector<Ipp32fc> data(100);
    ippe::generator::Slope(
        reinterpret_cast<Ipp32f*>(data.data()),
        (int)data.size()*2,
        0.0f,
        1.0f
    );
    for (int i = 0; i < data.size(); i++)
        printf("%f %f\n", data[i].re, data[i].im);

    // Instantiate the xcorr obj
    GroupXcorrCZT gxc(
        12, -50.0, 50.0, 50.0, 1000.0
    );
    int outCols = gxc.getCZTdimensions();
    printf("outCols = %d\n", outCols);

    // Add some groups
    gxc.addGroup(
        0, 10, &data.at(10), // use from 10 to 20 (exclusive), not the maxlength
        true
    );
    // Make sure error thrown for forward overlapping group
    try{
        gxc.addGroup(
            2, 10, &data.at(12), 
            true
        );
    }
    catch(std::range_error &e){
        printf("Expected range error: %s\n", e.what());
    }

    gxc.addGroup(
        60, 12, &data.at(70), // use from 70 to 82 (exclusive), max length
        true
    );
    // Make sure error thrown for backward overlapping group
    try{
        gxc.addGroup(
            50, 12, &data.at(60), 
            true
        );
    }
    catch(std::range_error &e){
        printf("Expected range error: %s\n", e.what());
    }
    // Also make sure error thrown for too long groups
    try{
        gxc.addGroup(
            40, 13, &data.at(50), 
            true
        );
    }
    catch(std::range_error &e){
        printf("Expected range error: %s\n", e.what());
    }

    // Print the groups
    gxc.printGroups();

    // Make the output
    int shiftStart = 9;
    int numShifts = 3;
    int shiftStep = 1;
    ippe::vector<Ipp32f> out(numShifts*outCols, 0.0f);
    // Run the xcorr
    gxc.xcorr(
        data.data(), shiftStart, shiftStep, numShifts, out.data()
    );

    // Display output
    for (int i = 0; i < numShifts; i++){
        for (int j = 0; j < outCols; j++){
            printf("%.6g  ", out.at(i*outCols + j));
        }
        printf("\n");
    }


    printf("COMPLETE\n");
    return 0;
}