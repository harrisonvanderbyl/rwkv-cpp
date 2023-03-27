# download libtorch libs

wget "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu117.zip"
unzip "libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117.zip"
rm "libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117.zip"

# copy libtorch libs to ./lib
# libc10_cuda.so  libcublas-f6acd947.so.11    libcudart-e409450e.so.11.0  libnvToolsExt-847d78f2.so.1  libtorch_cuda.so
# libc10.so       libcublasLt-2e7ea254.so.11  libgomp-52f2fd74.so.1       libtorch_cpu.so              libtorch.so
mkdir ./lib
cp "./libtorch/lib/libc10_cuda.so" "./lib"
cp "./libtorch/lib/libcublas-f6acd947.so.11" "./lib"
cp "./libtorch/lib/libcudart-e409450e.so.11.0" "./lib"
cp "./libtorch/lib/libnvToolsExt-847d78f2.so.1" "./lib"
cp "./libtorch/lib/libtorch_cuda.so" "./lib"
cp "./libtorch/lib/libc10.so" "./lib"
cp "./libtorch/lib/libcublasLt-2e7ea254.so.11" "./lib"
cp "./libtorch/lib/libgomp-52f2fd74.so.1" "./lib"
cp "./libtorch/lib/libtorch_cpu.so" "./lib"
cp "./libtorch/lib/libtorch.so" "./lib"

# remove libtorch
rm ./libtorch -r

# download model 
wget "https://huggingface.co/Hazzzardous/rwkv-fastquant/resolve/main/rwkv-7B-alpaca-2-1-2.rwkv"

