# CUDA_jacobianEigenvalue

`cuda-samples/Common`には，CUDAバージョンが一致するように，[cuda-samples](https://github.com/NVIDIA/cuda-samples)からダウンロードしたヘッダファイルを配置すること．

ビルド時には以下のコマンドを実行する.`DCMAKE_CUDA_ARCHITECTURES`には使用しているGPUのSMに合わせて変更すること．

```
mkdir build
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=89
cd build
make
```