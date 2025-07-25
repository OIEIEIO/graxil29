// src/gpu/kernels/opencl/cuckaroo29.cl - Cuckaroo29 OpenCL Mining Kernels
// Tree location: ./src/gpu/kernels/opencl/cuckaroo29.cl

//! Cuckaroo29 OpenCL implementation extracted from grin-miner
//! 
//! Original: Cuckaroo Cycle, a memory-hard proof-of-work by John Tromp and team Grin
//! Copyright (c) 2018 Jiri Photon Vadura and John Tromp
//! This GGM miner file is covered by the FAIR MINING license
//! 
//! Adapted for Graxil29 miner with multi-platform OpenCL support
//! 
//! # Version History
//! - 0.1.0: Initial extraction from grin-miner trimmer.rs SRC constant
//! - 0.1.1: Added comprehensive documentation and structure comments
//! 
//! # Algorithm Overview
//! 1. FluffySeed2A: Generate edges using SipHash from block header
//! 2. FluffySeed2B: Distribute edges into buckets for processing
//! 3. FluffyRound1: First trimming round with degree filtering
//! 4. FluffyRoundN*: 120+ trimming rounds to reduce graph
//! 5. FluffyTail: Collect remaining edges after trimming
//! 6. FluffyRecovery: Recover nonces from solution cycles
//! 
//! # Memory Layout
//! - Buffer A1: Primary edge storage (DUCK_SIZE_A * 1024 * 4096 * 2)
//! - Buffer A2: Secondary edge storage (DUCK_SIZE_A * 1024 * 256 * 2)  
//! - Buffer B:  Bucket storage (DUCK_SIZE_B * 1024 * 4096 * 2)
//! - Index buffers: Edge counting and bucket management

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

// Type definitions for clarity and compatibility
typedef uint8 u8;
typedef uint16 u16;
typedef uint u32;
typedef ulong u64;
typedef u32 node_t;
typedef u64 nonce_t;

// Cuckaroo29 algorithm constants
#define DUCK_SIZE_A 129L          // AMD optimized: 126 + 3
#define DUCK_SIZE_B 83L           // Bucket size for trimming
#define DUCK_A_EDGES (DUCK_SIZE_A * 1024L)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64L)
#define DUCK_B_EDGES (DUCK_SIZE_B * 1024L)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64L)

// Edge processing constants
#define EDGE_BLOCK_SIZE (64)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)

// Cuckaroo29 specific: 29-bit edge space
#define EDGEBITS 29
#define NEDGES ((node_t)1 << EDGEBITS)    // 2^29 = 536,870,912 edges
#define EDGEMASK (NEDGES - 1)             // Mask for 29-bit values

// Threading and bucket constants
#define CTHREADS 1024             // Compute threads per workgroup
#define BKTMASK4K (4096-1)        // 4K bucket mask
#define BKTGRAN 32                // Bucket granularity

// Advanced bucket constants for optimized rounds
#define BKT_OFFSET 255
#define BKT_STEP 32

//! SipHash round implementation
//! Core cryptographic function for edge generation
//! Performs the SipHash compression function with proper bit rotations
#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = rotate(v1,(ulong)13); \
    v3 = rotate(v3,(ulong)16); v1 ^= v0; v3 ^= v2; \
    v0 = rotate(v0,(ulong)32); v2 += v1; v0 += v3; \
    v1 = rotate(v1,(ulong)17);   v3 = rotate(v3,(ulong)21); \
    v1 ^= v2; v3 ^= v0; v2 = rotate(v2,(ulong)32); \
  } while(0)

//! 2-bit counter management for edge degree tracking
//! Increases counter and sets overflow bit if needed
void Increase2bCounter(__local u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;           // Divide by 32 for word index
	unsigned char bit = bucket & 0x1F; // Modulo 32 for bit position
	u32 mask = 1 << bit;

	u32 old = atomic_or(ecounters + word, mask) & mask;

	// Set overflow bit in second half if already set
	if (old > 0)
		atomic_or(ecounters + word + 4096, mask);
}

//! Read 2-bit counter overflow status
//! Returns true if edge should be kept (degree >= 2)
bool Read2bCounter(__local u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	return (ecounters[word + 4096] & mask) > 0;
}

//! KERNEL: FluffySeed2A - Initial edge generation using SipHash
//! 
//! Generates edges from block nonces using SipHash with header-derived keys.
//! This is the core edge generation that creates the bipartite graph.
//! 
//! @param v0i,v1i,v2i,v3i: SipHash keys derived from block header
//! @param bufferA: Primary edge buffer for first 32 buckets
//! @param bufferB: Secondary edge buffer for remaining buckets  
//! @param indexes: Edge count tracking per bucket
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel  void FluffySeed2A(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, 
                           __global ulong4 * bufferA, __global ulong4 * bufferB, 
                           __global u32 * indexes)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);

	__global ulong4 * buffer;
	__local u64 tmp[64][16];          // Local storage for edge batching
	__local u32 counters[64];         // Local counters per bucket
	u64 sipblock[64];                 // SipHash output block

	u64 v0, v1, v2, v3;               // SipHash state variables

	// Initialize local counters
	if (lid < 64)
		counters[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Process 2K nonces per work item in blocks of 64
	for (int i = 0; i < 1024 * 2; i += EDGE_BLOCK_SIZE)
	{
		u64 blockNonce = gid * (1024 * 2) + i;

		// Initialize SipHash state with header keys
		v0 = v0i; v1 = v1i; v2 = v2i; v3 = v3i;

		// Generate SipHash for block of nonces
		for (u32 b = 0; b < EDGE_BLOCK_SIZE; b++)
		{
			v3 ^= blockNonce + b;
			for (int r = 0; r < 2; r++)      // 2 SipRounds for compression
				SIPROUND;
			v0 ^= blockNonce + b;
			v2 ^= 0xff;                      // Finalization constant
			for (int r = 0; r < 4; r++)      // 4 SipRounds for finalization
				SIPROUND;

			sipblock[b] = (v0 ^ v1) ^ (v2  ^ v3);  // Final hash output
		}
		
		u64 last = sipblock[EDGE_BLOCK_MASK];

		// Extract edges and distribute to buckets
		for (short s = 0; s < EDGE_BLOCK_SIZE; s++)
		{
			u64 lookup = s == EDGE_BLOCK_MASK ? last : sipblock[s] ^ last;
			uint2 hash = (uint2)(lookup & EDGEMASK, (lookup >> 32) & EDGEMASK);
			int bucket = hash.x & 63;        // 64 buckets total

			barrier(CLK_LOCAL_MEM_FENCE);

			// Add edge to local bucket
			int counter = atomic_add(counters + bucket, (u32)1);
			int counterLocal = counter % 16;
			tmp[bucket][counterLocal] = hash.x | ((u64)hash.y << 32);

			barrier(CLK_LOCAL_MEM_FENCE);

			// Flush bucket when it reaches 8 or 16 edges
			if ((counter > 0) && (counterLocal == 0 || counterLocal == 8))
			{
				int cnt = min((int)atomic_add(indexes + bucket, 8), (int)(DUCK_A_EDGES_64 - 8));
				int idx = ((bucket < 32 ? bucket : bucket - 32) * DUCK_A_EDGES_64 + cnt) / 4;
				buffer = bucket < 32 ? bufferA : bufferB;

				// Write 8 edges as two ulong4 vectors
				buffer[idx] = (ulong4)(
					atom_xchg(&tmp[bucket][8 - counterLocal], (u64)0),
					atom_xchg(&tmp[bucket][9 - counterLocal], (u64)0),
					atom_xchg(&tmp[bucket][10 - counterLocal], (u64)0),
					atom_xchg(&tmp[bucket][11 - counterLocal], (u64)0)
				);
				buffer[idx + 1] = (ulong4)(
					atom_xchg(&tmp[bucket][12 - counterLocal], (u64)0),
					atom_xchg(&tmp[bucket][13 - counterLocal], (u64)0),
					atom_xchg(&tmp[bucket][14 - counterLocal], (u64)0),
					atom_xchg(&tmp[bucket][15 - counterLocal], (u64)0)
				);
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Flush remaining edges in local buckets
	if (lid < 64)
	{
		int counter = counters[lid];
		int counterBase = (counter % 16) >= 8 ? 8 : 0;
		int counterCount = (counter % 8);
		
		// Zero unused slots
		for (int i = 0; i < (8 - counterCount); i++)
			tmp[lid][counterBase + counterCount + i] = 0;
			
		int cnt = min((int)atomic_add(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));
		int idx = ( (lid < 32 ? lid : lid - 32) * DUCK_A_EDGES_64 + cnt) / 4;
		buffer = lid < 32 ? bufferA : bufferB;
		
		buffer[idx] = (ulong4)(tmp[lid][counterBase], tmp[lid][counterBase + 1], 
		                      tmp[lid][counterBase + 2], tmp[lid][counterBase + 3]);
		buffer[idx + 1] = (ulong4)(tmp[lid][counterBase + 4], tmp[lid][counterBase + 5], 
		                          tmp[lid][counterBase + 6], tmp[lid][counterBase + 7]);
	}
}

//! KERNEL: FluffySeed2B - Secondary bucketing and redistribution
//! 
//! Takes edges from FluffySeed2A and redistributes them into smaller buckets
//! for more efficient trimming. Implements a two-stage bucketing strategy.
//! 
//! @param source: Input edges from FluffySeed2A
//! @param destination1: Primary output buffer
//! @param destination2: Secondary output buffer  
//! @param sourceIndexes: Input edge counts
//! @param destinationIndexes: Output edge counts
//! @param startBlock: Which block (0 or 32) to process
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel  void FluffySeed2B(const __global uint2 * source, __global ulong4 * destination1, 
                           __global ulong4 * destination2, const __global int * sourceIndexes, 
                           __global int * destinationIndexes, int startBlock)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	__global ulong4 * destination = destination1;
	__local u64 tmp[64][16];
	__local int counters[64];

	if (lid < 64)
		counters[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Calculate memory offsets and bucket parameters
	int offsetMem = startBlock * DUCK_A_EDGES_64;
	int offsetBucket = 0;
	const int myBucket = group / BKTGRAN;
	const int microBlockNo = group % BKTGRAN;
	const int bucketEdges = min(sourceIndexes[myBucket + startBlock], (int)(DUCK_A_EDGES_64));
	const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN);
	const int loops = (microBlockEdgesCount / 128);

	// Handle buffer switching for higher buckets
	if ((startBlock == 32) && (myBucket >= 30))
	{
		offsetMem = 0;
		destination = destination2;
		offsetBucket = 30;
	}

	// Process edges in micro-blocks
	for (int i = 0; i < loops; i++)
	{
		int edgeIndex = (microBlockNo * microBlockEdgesCount) + (128 * i) + lid;

		{
			uint2 edge = source[(myBucket * DUCK_A_EDGES_64) + edgeIndex];
			bool skip = (edgeIndex >= bucketEdges) || (edge.x == 0 && edge.y == 0);

			int bucket = (edge.x >> 6) & (64 - 1);  // Extract bucket from high bits

			barrier(CLK_LOCAL_MEM_FENCE);

			int counter = 0;
			int counterLocal = 0;

			if (!skip)
			{
				counter = atomic_add(counters + bucket, (u32)1);
				counterLocal = counter % 16;
				tmp[bucket][counterLocal] = edge.x | ((u64)edge.y << 32);
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			// Flush when bucket reaches capacity
			if ((counter > 0) && (counterLocal == 0 || counterLocal == 8))
			{
				int cnt = min((int)atomic_add(destinationIndexes + startBlock * 64 + myBucket * 64 + bucket, 8), 
				             (int)(DUCK_A_EDGES - 8));
				int idx = (offsetMem + (((myBucket - offsetBucket) * 64 + bucket) * DUCK_A_EDGES + cnt)) / 4;

				destination[idx] = (ulong4)(
					atom_xchg(&tmp[bucket][8 - counterLocal], 0),
					atom_xchg(&tmp[bucket][9 - counterLocal], 0),
					atom_xchg(&tmp[bucket][10 - counterLocal], 0),
					atom_xchg(&tmp[bucket][11 - counterLocal], 0)
				);
				destination[idx + 1] = (ulong4)(
					atom_xchg(&tmp[bucket][12 - counterLocal], 0),
					atom_xchg(&tmp[bucket][13 - counterLocal], 0),
					atom_xchg(&tmp[bucket][14 - counterLocal], 0),
					atom_xchg(&tmp[bucket][15 - counterLocal], 0)
				);
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Final flush of remaining edges
	if (lid < 64)
	{
		int counter = counters[lid];
		int counterBase = (counter % 16) >= 8 ? 8 : 0;
		int cnt = min((int)atomic_add(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), 
		             (int)(DUCK_A_EDGES - 8));
		int idx = (offsetMem + (((myBucket - offsetBucket) * 64 + lid) * DUCK_A_EDGES + cnt)) / 4;
		destination[idx] = (ulong4)(tmp[lid][counterBase], tmp[lid][counterBase + 1], 
		                           tmp[lid][counterBase + 2], tmp[lid][counterBase + 3]);
		destination[idx + 1] = (ulong4)(tmp[lid][counterBase + 4], tmp[lid][counterBase + 5], 
		                               tmp[lid][counterBase + 6], tmp[lid][counterBase + 7]);
	}
}

//! KERNEL: FluffyRound1 - First trimming round with degree filtering
//! 
//! Implements the first round of graph trimming. Counts edge degrees and
//! keeps only edges with degree >= 2. This dramatically reduces the graph size.
//! 
//! @param source1: First input buffer
//! @param source2: Second input buffer  
//! @param destination: Output buffer for kept edges
//! @param sourceIndexes: Input edge counts
//! @param destinationIndexes: Output edge counts
//! @param bktInSize: Input bucket size
//! @param bktOutSize: Output bucket size
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel   void FluffyRound1(const __global uint2 * source1, const __global uint2 * source2, 
                            __global uint2 * destination, const __global int * sourceIndexes, 
                            __global int * destinationIndexes, const int bktInSize, const int bktOutSize)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	// Select source buffer based on group ID
	const __global uint2 * source = group < (62 * 64) ? source1 : source2;
	int groupRead                 = group < (62 * 64) ? group : group - (62 * 64);

	__local u32 ecounters[8192];      // 2-bit counters for edge degrees

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

	// Initialize edge counters
	for (int i = 0; i < 8; i++)
		ecounters[lid + (1024 * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// First pass: count edge degrees
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * groupRead) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Second pass: keep edges with degree >= 2
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * groupRead) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomic_add(destinationIndexes + bucket, 1), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}
}

//! KERNEL: FluffyRoundN - Standard trimming rounds
//! 
//! Implements the main trimming algorithm. Runs for ~120 iterations to
//! progressively remove low-degree nodes until only cycles remain.
//! 
//! @param source: Input edge buffer
//! @param destination: Output edge buffer
//! @param sourceIndexes: Input edge counts  
//! @param destinationIndexes: Output edge counts
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel   void FluffyRoundN(const __global uint2 * source, __global uint2 * destination, 
                            const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	const int bktInSize = DUCK_B_EDGES;
	const int bktOutSize = DUCK_B_EDGES;

	__local u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

	for (int i = 0; i < 8; i++)
		ecounters[lid + (1024 * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Count degrees
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Keep edges with degree >= 2
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomic_add(destinationIndexes + bucket, 1), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}
}

//! KERNEL: FluffyRoundNO1 - Optimized trimming round variant 1
//! 
//! Memory-optimized version of trimming with advanced bucket offsetting
//! for better memory access patterns on certain GPU architectures.
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel   void FluffyRoundNO1(const __global uint2 * source, __global uint2 * destination, 
                              const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	const int bktInSize = DUCK_B_EDGES;
	const int bktOutSize = DUCK_B_EDGES;

	__local u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

	for (int i = 0; i < 8; i++)
		ecounters[lid + (1024 * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Degree counting phase
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index =(bktInSize * group) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Edge keeping phase with memory optimization
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomic_add(destinationIndexes + bucket, 1), 
				                      bktOutSize - 1 - ((bucket & BKT_OFFSET) * BKT_STEP));
				destination[((bucket & BKT_OFFSET) * BKT_STEP) + (bucket * bktOutSize) + bktIdx] = 
				           (uint2)(edge.y, edge.x);
			}
		}
	}
}

//! KERNEL: FluffyRoundNON - Optimized trimming round variant 2  
//! 
//! Alternative memory layout optimization for the trimming rounds.
//! Uses different bucket addressing for improved cache efficiency.
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel   void FluffyRoundNON(const __global uint2 * source, __global uint2 * destination, 
                              const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	const int bktInSize = DUCK_B_EDGES;
	const int bktOutSize = DUCK_B_EDGES;

	__local u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

	for (int i = 0; i < 8; i++)
		ecounters[lid + (1024 * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Count degrees with offset addressing
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = ((group & BKT_OFFSET) * BKT_STEP) + (bktInSize * group) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Keep edges with offset addressing
	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = ((group & BKT_OFFSET) * BKT_STEP) + (bktInSize * group) + lindex;
			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0) continue;

			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomic_add(destinationIndexes + bucket, 1), 
				                      bktOutSize - 1 - ((bucket & BKT_OFFSET) * BKT_STEP));
				destination[((bucket & BKT_OFFSET) * BKT_STEP) + (bucket * bktOutSize) + bktIdx] = 
				           (uint2)(edge.y, edge.x);
			}
		}
	}
}

//! KERNEL: FluffyTail - Final edge collection
//! 
//! Collects remaining edges after all trimming rounds into a compact array.
//! This produces the final edge set that will be searched for cycles.
//! 
//! @param source: Input edges from final trimming round
//! @param destination: Compact output array of remaining edges
//! @param sourceIndexes: Number of edges per bucket
//! @param destinationIndexes: Global counter for total edges
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void FluffyTail(const __global uint2 * source, __global uint2 * destination, 
                        const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	int myEdges = sourceIndexes[group];
	__local int destIdx;

	// Get global destination index for this bucket
	if (lid == 0)
		destIdx = atomic_add(destinationIndexes, myEdges);

	barrier(CLK_LOCAL_MEM_FENCE);

	// Copy edges to compact array
	if (lid < myEdges)
	{
		destination[destIdx + lid] = source[group * DUCK_B_EDGES + lid];
	}
}

//! KERNEL: FluffyTailO - Optimized tail collection
//! 
//! Memory-optimized version of FluffyTail with advanced addressing.
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void FluffyTailO(const __global uint2 * source, __global uint2 * destination, 
                         const __global int * sourceIndexes, __global int * destinationIndexes)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	int myEdges = sourceIndexes[group];
	__local int destIdx;

	if (lid == 0)
		destIdx = atomic_add(destinationIndexes, myEdges);

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < myEdges)
	{
		destination[destIdx + lid] = source[((group & BKT_OFFSET) * BKT_STEP) + group * DUCK_B_EDGES + lid];
	}
}

//! KERNEL: FluffyRecovery - Nonce recovery from solution cycles
//! 
//! Given a valid 42-cycle from the graph search, recovers the original nonces
//! that generated those edges. This verifies the solution and provides the
//! proof-of-work nonces for block submission.
//! 
//! @param v0i,v1i,v2i,v3i: SipHash keys (same as generation)
//! @param recovery: Array of 42 edges from the found cycle  
//! @param indexes: Output array of 42 nonces that generated the cycle
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel   void FluffyRecovery(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, 
                              const __constant u64 * recovery, __global int * indexes)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);

	__local u32 nonces[42];
	u64 sipblock[64];
	u64 v0, v1, v2, v3;

	// Initialize nonce array
	if (lid < 42) 
		nonces[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Search through nonce space to find matches
	for (int i = 0; i < 1024; i += EDGE_BLOCK_SIZE)
	{
		u64 blockNonce = gid * 1024 + i;

		// Generate SipHash block (same as FluffySeed2A)
		v0 = v0i; v1 = v1i; v2 = v2i; v3 = v3i;

		for (u32 b = 0; b < EDGE_BLOCK_SIZE; b++)
		{
			v3 ^= blockNonce + b;
			SIPROUND; SIPROUND;                // 2 compression rounds
			v0 ^= blockNonce + b;
			v2 ^= 0xff;                        // Finalization
			SIPROUND; SIPROUND; SIPROUND; SIPROUND;  // 4 finalization rounds

			sipblock[b] = (v0 ^ v1) ^ (v2 ^ v3);
		}
		
		const u64 last = sipblock[EDGE_BLOCK_MASK];

		// Check each edge in block against recovery set
		for (short s = EDGE_BLOCK_MASK; s >= 0; s--)
		{
			u64 lookup = s == EDGE_BLOCK_MASK ? last : sipblock[s] ^ last;
			u64 u = lookup & EDGEMASK;
			u64 v = (lookup >> 32) & EDGEMASK;

			// Create both edge orientations
			u64 a = u | (v << 32);
			u64 b = v | (u << 32);

			// Check against all 42 cycle edges
			for (int i = 0; i < 42; i++)
			{
				if ((recovery[i] == a) || (recovery[i] == b))
					nonces[i] = blockNonce + s;
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Output found nonces
	if (lid < 42)
	{
		if (nonces[lid] > 0)
			indexes[lid] = nonces[lid];
	}
}