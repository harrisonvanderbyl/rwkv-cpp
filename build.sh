nvcc -c ./cudaops/operators.cu ./cudaops/operators32.cu -I$(pwd)/libtorch/include
cmake ./CMakeLists.txt 
cmake -DCMAKE_PREFIX_PATH=$(pwd)/libtorch .
mkdir release
cmake --build . --config Debug

mv ./RWKVCPP ./release/RWKVCPP 
# if lib already exists, do not copy
if [ ! -d "./release/lib" ]; then
    cp ./libtorch/lib/ ./release/ -r
fi
