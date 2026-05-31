#include "blake3_tree_host.hpp"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
extern "C" void pearl_blake3_chunk_cvs_sm89(const uint8_t*, size_t, const uint8_t[32], uint32_t*, cudaStream_t);
int main(){
  FILE*f=fopen("/tmp/ha.bin","rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<uint8_t> data(n); if(fread(data.data(),1,n,f)!=(size_t)n) return 2; fclose(f);
  uint8_t key[32]; for(int i=0;i<32;i++) key[i]=(uint8_t)(i*7+1);
  size_t nch = n/1024;
  uint8_t* dd; cudaMalloc(&dd,n); cudaMemcpy(dd,data.data(),n,cudaMemcpyHostToDevice);
  uint32_t* dcv; cudaMalloc(&dcv, nch*32);
  pearl_blake3_chunk_cvs_sm89(dd, n, key, dcv, 0);
  if(cudaDeviceSynchronize()!=cudaSuccess){printf("kernel err %s\n",cudaGetErrorString(cudaGetLastError()));return 2;}
  std::vector<uint32_t> cvs(nch*8); cudaMemcpy(cvs.data(), dcv, nch*32, cudaMemcpyDeviceToHost);
  uint8_t gpuroot[32], refroot[32];
  pearl_miner::b3tree::blake3_root_from_chunk_cvs(key, cvs.data(), nch, gpuroot);
  pearl_miner::b3tree::blake3_keyed_tree(key, data.data(), n, refroot);
  printf("GPUROOT "); for(int i=0;i<32;i++) printf("%02x",gpuroot[i]); printf("\n");
  printf("REFROOT "); for(int i=0;i<32;i++) printf("%02x",refroot[i]); printf("\n");
  return 0;
}
