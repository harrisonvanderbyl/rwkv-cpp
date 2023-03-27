nvcc -c ./cudaops/operators.cu ./cudaops/operators32.cu -I$(pwd)/libtorch/include
cmake ./CMakeLists.txt 
cmake -DCMAKE_PREFIX_PATH=$(pwd)/libtorch .
mkdir release
cmake --build . --config Debug

mv ./RWKVCPP ./release/RWKVCPP 

# ask if full bundle
read -p "Do you want to bundle the full release? (y/n) " -n 1 -r

# if yes, cp libs from libtorch
if [[ $REPLY =~ ^[Yy]$ ]]
then
    mkdir ./release/lib
    cp "./libtorch/lib/libc10_cuda.so" "./release/lib"
    cp "./libtorch/lib/libcublas-f6acd947.so.11" "./release/lib"
    cp "./libtorch/lib/libcudart-e409450e.so.11.0" "./release/lib"
    cp "./libtorch/lib/libnvToolsExt-847d78f2.so.1" "./release/lib"
    cp "./libtorch/lib/libtorch_cuda.so" "./release/lib"
    cp "./libtorch/lib/libc10.so" "./release/lib"
    cp "./libtorch/lib/libcublasLt-2e7ea254.so.11" "./release/lib"
    cp "./libtorch/lib/libgomp-52f2fd74.so.1" "./release/lib"
    cp "./libtorch/lib/libtorch_cpu.so" "./release/lib"
    cp "./libtorch/lib/libtorch.so" "./release/lib"
fi

# zip release
cd release
zip -r release.zip ./*
cd ..

# copy zip to bin folder
cp ./release/release.zip ./bin