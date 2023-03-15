cmake ./CMakeLists.txt 
cmake -DCMAKE_PREFIX_PATH=$(pwd)/libtorch .
mkdir release
cmake --build . --config Debug

mv ./RWKVCPP ./release/RWKVCPP 
cp ./libtorch/lib/ ./release/lib/ -r