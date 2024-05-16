cd CyGroupXcorrFFT
python setup_GroupXcorrFFT.py build_ext --inplace
cd ..

cd CyIppXcorrFFT
python setup_IppXcorrFFT.py build_ext --inplace
cd ..

cd PySampledLinearInterpolator
python setup_SampledLinearInterpolator.py build_ext --inplace
cd ..

cd PyViterbiDemodulator
python setup_ViterbiDemodulator.py build_ext --inplace
cd ..

cd compareIntPreambles
compile.bat
cd ..
