
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride) 
{
	// TO DO: Kernel implementation

	//1st method in comments was not efficient since it skipped some of the work items (idle threads) because of the if clause
	/*int GID = get_global_id(0);
	if ((GID % (2*stride)) == 0)
	{
		printf("stride(%d): [%d]+[%d] = %d + %d\n", stride, GID, GID + stride, array[GID], array[GID + stride]);
		array[GID] += array[GID + stride];
	}*/
	
	int GID = get_global_id(0);
	int IDx = GID * 2 * stride;
	array[IDx] += array[IDx + stride];
	//printf("stride(%d) GID(%d): [%d]+[%d] = %d + %d = %d\n", stride, GID, IDx, IDx + stride, array[GID]-array[GID + offset], array[IDx + stride], array[GID]);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride) 
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	int offset = get_local_size(0); //the offset corresponds to the size of a group
	array[GID] += array[GID + offset];
	//printf("stride(%d) GID(%d): [%d]+[%d] = %d + %d = %d\n", stride, GID, GID, GID + offset, array[GID]-array[GID + offset], array[GID + offset], array[GID]);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	int offset = get_local_size(0);

	//printf("GID(%d): localmem [%d] = in [%d] + in [%d] = %d + %d\n", GID,  LID, GID, GID+offset, inArray[GID], inArray[GID + offset]);

	localBlock[LID] = inArray[GID] + inArray[GID + offset];
	barrier(CLK_LOCAL_MEM_FENCE);

	//area of local memory divided by 2
	uint localOffset = offset / 2;

	for (localOffset; localOffset >= 1; localOffset /= 2) //area of local memory divided by 2
	{
		if (LID < localOffset)
		{
			//printf("localmem area: %d\n", localOffset);
			//printf("GID(%d): localmem [%d] = localmem [%d] + localmem [%d] = %d + %d\n", GID, LID, LID, LID + localOffset, localBlock[LID], localBlock[LID + localOffset]);
			localBlock[LID] += localBlock[LID + localOffset];
			//printf("GID(%d): localmem [%d] = %d\n", GID, LID, localBlock[LID]);
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	//writes from local shared memory to global device memory
	if (LID == 0) {
		outArray[get_group_id(0)] = localBlock[0];
		//printf("GID(%d): out [] = %d\n", GID, localBlock[LID]);
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	int offset = get_local_size(0);

	localBlock[LID] = inArray[GID + offset] + inArray[GID];
	barrier(CLK_LOCAL_MEM_FENCE);

	uint localOffset = offset / 2;

	__attribute__((opencl_unroll_hint)) //specify that a loop can be unrolled
	for (localOffset; localOffset > 1; localOffset /= 2)  //Halve area
	{
		if (LID < localOffset) 
		{
			//printf("%d\n", localOffset);
			//printf("GID (%d): idx [%d] + [%d] = %d + %d with stride %d \n",GID,  LID, LID+localOffset, localBlock[LID], localBlock[LID+localOffset], localOffset);
			localBlock[LID] += localBlock[LID + localOffset];

			barrier(CLK_LOCAL_MEM_FENCE);

		}
	}

	//Write back
	if (LID == 0) {
		localBlock[0] += localBlock[1];
		outArray[get_group_id(0)] = localBlock[0];
	}
	//if (LID == 0) printf("%d\n", localBlock[0]);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompAtomics(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localSum)
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	int offset = get_local_size(0);
	int groupID = get_group_id(0);

	*localSum = 0;

	atom_add(&localSum[0], inArray[GID]);
	atom_add(&localSum[0], inArray[GID + offset]);

	barrier(CLK_LOCAL_MEM_FENCE);

	if (LID == 0) outArray[groupID] = localSum[0];
}
