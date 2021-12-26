


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) 
{
	// TO DO: Kernel implementation
    uint GID = get_global_id(0);

	if (GID > N) //out of bounds
	{
		return;
	}
    else if (GID >= offset)//after the offse -> perform add operation
	{
        outArray[GID] = inArray[GID] + inArray[GID - offset];
		//printf("GID(%d): out [%d] = in [%d] + in [%d] = %d + %d = %d\n", GID, GID, GID, GID - offset, inArray[GID], inArray[GID - offset], outArray[GID]);
    }
    else if (GID < offset)//before the offset -> value stays the same
	{
        outArray[GID] = inArray[GID];
		//printf("GID(%d): out [%d] = in [%d] = %d = %d\n", GID, GID, GID, inArray[GID], outArray[GID]);
    }
}

// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
	#define OFFSET(A) (((A)/NUM_BANKS) + (A)) 
#else
	#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) 
{
	// TO DO: Kernel implementation
	uint GID = get_global_id(0);
	uint LID = get_local_id(0);
	uint localSize = get_local_size(0);
	uint globalSize = get_global_size(0);
	uint groupID = get_group_id(0);

	//store in a 2N elements local memory block the values of the input array
	//each thread stores 2 elements

	//localBlock[LID * 2] = array[GID * 2];
	//localBlock[LID * 2 + 1] = array[GID * 2 + 1];

	localBlock[OFFSET(LID)] = array[LID + 2*localSize*groupID];
	localBlock[OFFSET(LID + localSize)] = array[LID * 2 * localSize * groupID + localSize];

	barrier(CLK_LOCAL_MEM_FENCE);

	// Up-Sweep
	for (uint stride = 1; stride*2 <= localSize*2; stride *= 2)
	{
		if (LID < localSize/(2*stride)) 
		{
			//uint index1 = LID * (2 * stride) + stride * 2 - 1;
			//uint index2 = LID * (2 * stride) + stride - 1;

			uint index1 = OFFSET(localSize - LID * (2 * stride) - 1);
			uint index2 = OFFSET(localSize - LID * (2 * stride) - stride - 1);

			localBlock[index1] += localBlock[index2];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//if (GID == 0) localBlock[localSize * 2 - 1] = 0;

	localBlock[OFFSET(localSize - 1)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Down-Sweep
	for (uint stride = localSize; stride >= 1; stride /= 2)
	{
		if (LID < localSize/stride) 
		{
			//uint index1 = LID * (2 * stride) + stride * 2 - 1;
			//uint index2 = LID * (2 * stride) + stride - 1;
			//uint val1 = localBlock[index1];
			//uint val2 = localBlock[index2] + localBlock[index1];
			//localBlock[index2] = val1;
			//localBlock[index1] = val2;

			uint index1 = OFFSET(localSize * 2 - LID * (2 * stride) - 1);
			uint index2 = OFFSET(localSize * 2 - LID * (2 * stride) - stride - 1);

			uint val = localBlock[index1];
			localBlock[index1] += localBlock[index2];
			localBlock[index2] = val;
			
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//array[GID * 2] = array[GID * 2] + localBlock[LID * 2];
	//array[GID * 2 + 1] = array[GID * 2 + 1] + localBlock[LID * 2 + 1];
	array[LID + 2 * localSize * groupID] += localBlock[OFFSET(LID)];
	array[localSize + LID + 2 * localSize * groupID] += localBlock[OFFSET(localSize + LID)];

	//if (GID == 0) printf("GID %d: %d \n", GID, localBlock[GID]);

	//load from
	if (LID == localSize - 1)
	{
		higherLevelArray[groupID] = array[LID + 2*localSize * groupID + localSize];
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock) 
{
	// TO DO: Kernel implementation (large arrays)
	// Kernel that should add the group PPS to the local PPS (Figure 14)
	uint GID = get_global_id(0);
	uint groupID = get_group_id(0);

	if ((groupID / 2 - 1) < 0)
	{
		return;
	}
	
	array[GID] += higherLevelArray[groupID / 2 - 1];
}