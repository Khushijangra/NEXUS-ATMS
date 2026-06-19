# Hardware Execution Report

## Phase C — Hardware Feasibility

### Target Hardware Configuration
* **GPU**: NVIDIA RTX 2050
* **VRAM**: 4GB
* **Environment**: CUDA 12.4, PyTorch 2.6

### 1. Can VideoMAE feature extraction run locally?
**Yes**, but *only under strict constraints*. VideoMAE-v2 inference uses between 2.0GB and 3.5GB of VRAM depending on the model scale (Base vs Large). To run on 4GB VRAM, the extraction script must operate strictly sequentially with a batch size of `1`, and no other models (e.g., D3QN, SUMO GUI) can be loaded into VRAM concurrently.

### 2. Expected Throughput
Assuming a batch size of 1 and standard 16-frame clip chunking:
* **clips/minute**: ~10 - 15 clips per minute (dependent on I/O bottlenecks).
* **videos/hour**: For a standard 10-minute traffic video (chunked into hundreds of clips), expect ~1-2 videos per hour.

### 3. VRAM Requirements
* **Peak Memory Allocation**: ~3.2 GB during the forward pass.
* **Overhead buffer**: Leaves ~0.8 GB for Windows Desktop Window Manager and PyTorch context. 

### 4. System RAM Requirements
* **Minimum**: 16 GB RAM recommended. Decord or OpenCV video decoding will heavily tax system RAM as frames are pre-fetched and batched before passing to the GPU.

### 5. Disk Requirements
* **Raw Video Storage**: 50 GB - 250 GB (dependent on chosen dataset).
* **Feature Storage (`.npy`)**: 10 GB - 30 GB. Extracted `[16, 768]` temporal embeddings consume significantly less space than raw MP4 files but still require fast SSD access for subsequent MULDE training.

**Conclusion**: Extraction is feasible, but will be a significant temporal bottleneck requiring dedicated overnight processing.
