mkdir build
g++ CZT.cpp testCZT.cpp -lippcore -lipps -I. -I../../ipp_ext/include -O3 -o build/testCZT
