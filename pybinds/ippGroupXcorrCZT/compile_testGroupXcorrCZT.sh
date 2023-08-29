mkdir build
g++ GroupXcorrCZT.cpp testGroupXcorrCZT.cpp ../ippCZT/CZT.cpp -lippcore -lipps -I../ippCZT/ -I../../ipp_ext/include -O3 -Wall -o build/testGroupXcorrCZT -DNDEBUG