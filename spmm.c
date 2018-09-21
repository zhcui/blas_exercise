#include <stdio.h>
#include "mkl.h"

extern "C" void spmm(const int & M, const int & N, const int & K, 
		int* rowIndex_A, int* columns_A, double* values_A,
		int* rowIndex_B, int* columns_B, double* values_B,
		int*& pointerB_C, int*& pointerE_C, int*& columns_C, double*& values_C, int & nnz){
    
	/* 
	 * Calculate C = A * B, where shape of A is M * K, B is K * N.
	 * All matrices are in CSR format, i.e. can be expressed with (compressed)rowIndex, columns and values.
	 * where length of rowIndex is number of row + 1, while length columns and values is number of non-zero elements,
	 * nnz.
	 * pointerB_C / pointerE_C is collection of begin / end index of each row of C.
	 *
	 * Since we usually do NOT know nnz of C before the calculation, we return it as well.
	 * 
	 * See https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-csr-matrix-storage-format
	 * for details.
	 *
	*/

	int  rows, cols;
    sparse_index_base_t    indexing;
    sparse_matrix_t        csrA = NULL, csrB = NULL, csrC = NULL;
    
	int i, j, ii;

	// print info of A 
	printf( "\n MATRIX A:\nrow# : (value, column) (value, column)\n" );
    ii = 0;
	for( i = 0; i < M; i++ )
    {
        printf("row#%d:", i + 1); fflush(0);
        for( j = rowIndex_A[i]; j < rowIndex_A[i+1]; j++ )
        {
            printf(" (%10.5f, %6d)", values_A[ii], columns_A[ii] ); fflush(0);
            ii++;
        }
        printf( "\n" );
    }

	// print info of B
    printf( "\n MATRIX B:\nrow# : (value, column)\n" );
    ii = 0;
    for( i = 0; i < M; i++ )
    {
        printf("row#%d:", i + 1); fflush(0);
        for( j = rowIndex_B[i]; j < rowIndex_B[i+1]; j++ )
        {
            printf(" (%10.5f, %6d)", values_B[ii], columns_B[ii] ); fflush(0);
            ii++;
        }
        printf( "\n" );
    }

	// create CSR handle of A and B
	mkl_sparse_d_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, M, K, rowIndex_A, rowIndex_A+1, columns_A, values_A );
    mkl_sparse_d_create_csr( &csrB, SPARSE_INDEX_BASE_ZERO, K, N, rowIndex_B, rowIndex_B+1, columns_B, values_B );

	// do multiplication and export the info.
    mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC );
    mkl_sparse_d_export_csr( csrC, &indexing, &rows, &cols, &pointerB_C, &pointerE_C, &columns_C, &values_C );
	nnz = pointerE_C[M - 1];
    

	// print the info of C
	printf( "\n RESULTANT MATRIX C:\nrow# : (value, column) (value, column)\n" );
    
	ii = 0;
    for( i = 0; i < M; i++ )
    {
        printf("row#%d:", i + 1); fflush(0);
        for( j = pointerB_C[i]; j < pointerE_C[i]; j++ )
        {
            printf(" (%10.5f, %6d)", values_C[ii], columns_C[ii] ); fflush(0);
            ii++;
        }
        printf( "\n" );
    }
	printf("nnz %d \n", nnz);
    printf( "_____________________________________________________________________  \n" );


	
	//destory the handlde
	mkl_sparse_destroy( csrA );
	mkl_sparse_destroy( csrB );
	mkl_sparse_destroy( csrC );
}
