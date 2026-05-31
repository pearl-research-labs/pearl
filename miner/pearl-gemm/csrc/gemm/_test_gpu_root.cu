#include "blake3_tree_host.hpp"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
extern "C" void pearl_blake3_chunk_cvs_sm89(const uint8_t*, size_t, const uint8_t[32], uint32_t*, cudaStream_t);
int main(){
  uint8_t key[32]; for(int i=0;i<32;i++) key[i]=(uint8_t)(i*7+1);
  // ---- correctness on /tmp/ha.bin ----
  { FILE*f=fopen("/tmp/ha.bin","rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint8_t> d(n); if(fread(d.data(),1,n,f)!=(size_t)n) return 2; fclose(f);
    size_t nch=n/1024; uint8_t* dd; cudaMalloc(&dd,n); cudaMemcpy(dd,d.data(),n,cudaMemcpyHostToDevice);
    uint32_t* dcv; cudaMalloc(&dcv,nch*32); pearl_blake3_chunk_cvs_sm89(dd,n,key,dcv,0); cudaDeviceSynchronize();
    std::vector<uint32_t> cvs(nch*8); cudaMemcpy(cvs.data(),dcv,nch*32,cudaMemcpyDeviceToHost);
    uint8_t gr[32], rr[32];
    pearl_miner::b3tree::blake3_root_from_chunk_cvs(key,cvs.data(),nch,gr);
    pearl_miner::b3tree::blake3_keyed_tree(key,d.data(),n,rr);
    bool ok=true; for(int i=0;i<32;i++) ok&=(gr[i]==rr[i]);
    printf("CORRECTNESS %s\n", ok?"MATCH":"MISMATCH");
    cudaFree(dd); cudaFree(dcv);
  }
  // ---- timing at the real matrix size: 131072*4096 = 512 MiB ----
  size_t N=(size_t)131072*4096; size_t nch=N/1024;
  uint8_t* dd; if(cudaMalloc(&dd,N)!=cudaSuccess){printf("oom\n");return 2;} cudaMemset(dd,7,N);
  uint32_t* dcv; cudaMalloc(&dcv,nch*32);
  pearl_blake3_chunk_cvs_sm89(dd,N,key,dcv,0); cudaDeviceSynchronize(); // warmup
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); for(int i=0;i<20;i++) pearl_blake3_chunk_cvs_sm89(dd,N,key,dcv,0); cudaEventRecord(b);
  cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b);
  printf("CHUNK_CVS 512MiB: %.2f ms/call (%d chunks)\n", ms/20.0, (int)nch);
  return 0;
}
