#pragma once

// Enum of NamedBarriers to avoid conflicts
// These are CTA-wide barriers with a modifiable arrival count, wrapping PTX "bar" instructions
// There are 16, of which CUTLASS currently reserves 8. We need more than 8 so we cast to
// CUTLASS's reserved barrier type in the kernels.
// CUTLASS recommends not to use barrier id 0 as this can conflict with CUDA driver APIs.
//

namespace pearl {
enum class NamedBarriers {
  MmaComplete = 1,
  DenoiseComplete = 2,
  DenoiseConvertEBRRead = 3,
  DenoiseConvertAxEBLRead = 4,
  DenoiseConvertAxEBWrite = 5,
  DenoiseConvertEARxBpEBRead = 6,
  DenoiseConvertEALRead = 7,
  DenoiseConvertEAxBpEBWrite = 8,
  Epilogue = 9,
  LoadScales = 10,
  // Path 3 (Blackwell consumer SMEM-fit): signals that WGMMA1's consumer
  // has released phase-1 denoise SMEM (EAL + EARxBpEB), so the producer
  // may safely overwrite the aliased SMEM with phase-2 (AxEBL + EBR) TMA
  // loads. Arrival count: kNumMmaThreads + NumThreadsPerWarp.
  DenoisePhase1Consumed = 11,
};
}
