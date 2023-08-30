mkdir build
g++ GroupXcorrCZT.cpp testGroupXcorrCZT.cpp ../ippCZT/CZT.cpp -lippcore -lipps -lpthread -I../ippCZT/ -I../../ipp_ext/include -O3 -o build/testGroupXcorrCZT -DNDEBUG