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

    // Instantiate the xcorr obj
    GroupXcorrCZT gxc(
        12, -0.1, 0.1, 0.1, 100.0
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
    gxc.xcorrRaw(
        data.data(), shiftStart, shiftStep, numShifts, out.data()
    );

    // Display output
    for (int i = 0; i < numShifts; i++){
        for (int j = 0; j < outCols; j++){
            printf("%.6g  ", out.at(i*outCols + j));
        }
        printf("\n");
    }

    // Try to xcorr with threads instead
    out.zero();
    printf("Running with threads\n");
    GroupXcorrCZT gxc4(
        12, -0.1, 0.1, 0.1, 100.0, 4
    );
    printf("Instantiated\n");
    try{
        gxc4.xcorrRaw(
            data.data(), shiftStart, shiftStep, numShifts, out.data()
        );
    }
    catch(std::exception &e){
        printf("Expected exception: %s\n", e.what());
    }

    gxc4.addGroup(
        0, 10, &data.at(10), // use from 10 to 20 (exclusive), not the maxlength
        true
    );
    gxc4.addGroup(
        60, 12, &data.at(70), // use from 70 to 82 (exclusive), max length
        true
    );
    gxc4.addGroup(
        10, 12, &data.at(20), // use from 20 to 32 (exclusive), max length
        true
    );
    gxc4.addGroup(
        30, 10, &data.at(40), // use from 40 to 52 (exclusive), max length
        true
    );
    try{
        gxc4.xcorrRaw(
            data.data(), shiftStart, shiftStep, numShifts, out.data()
        );
    }
    catch(std::exception &e){
        printf("This should be no exception: %s\n", e.what());
    }
    
    
    // Display output again
    for (int i = 0; i < numShifts; i++){
        for (int j = 0; j < outCols; j++){
            printf("%.6g  ", out.at(i*outCols + j));
        }
        printf("\n");
    }

    // Add a group that is way out of range
    gxc4.addGroup(
        100, 10, &data.at(10), true
    );

    try{
        gxc4.xcorrRaw(
            data.data(), shiftStart, shiftStep, numShifts, out.data(), data.size()
        );
    }
    catch(std::exception &e){
        printf("Expected exception: %s\n", e.what());
    }

    // Use the convenience multi group adder method for safety
    int starts[4] = {10, 70, 20, 40};
    int lengths[4] = {10, 12, 12, 10};
    gxc4.resetGroups();
    gxc4.addGroupsFromArray(
        starts, lengths, 4, data.data()
    );
    out.zero();
    try{
        gxc4.xcorrRaw(
            data.data(), shiftStart, shiftStep, numShifts, out.data(), data.size()
        );
    }
    catch(std::exception &e){
        printf("No exception expected\n");
    }

    // Display output again (should be identical to before)
    for (int i = 0; i < numShifts; i++){
        for (int j = 0; j < outCols; j++){
            printf("%.6g  ", out.at(i*outCols + j));
        }
        printf("\n");
    }


    printf("COMPLETE\n");
    return 0;
}