//nvcc -o libgpu_utils.so gpu_utils.cu -shared -Xcompiler -fPIC -lcublas # -lgomp


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"



extern "C" {
void alloc_arr(long *ptr, long nbytes, int *failed) {
  void *tmp;
  if (cudaMalloc(&tmp,nbytes)!=cudaSuccess) {
    fprintf(stderr,"error in cudaMalloc\n");
    failed[0]=1;
  }
  else
    failed[0]=0;
  ptr[0]=(long)tmp;
}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void togpu(long *dest, void *src, long nbytes, int *failed)
{
  void *tmp=(void *)(dest[0]);
  //printf("going to copy %ld bytes.\n",nbytes);
  if (cudaMemcpy(tmp,src,nbytes,cudaMemcpyHostToDevice)!=cudaSuccess) {
    fprintf(stderr,"Error copying data to device.\n");
    failed[0]=1;
  }
  else
    failed[0]=0;

}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void fromgpu(void *dest, long *src, long nbytes, int *failed)
{
  void *tmp=(void *)(src[0]);
  if (cudaMemcpy(dest,tmp,nbytes,cudaMemcpyDeviceToHost)!=cudaSuccess) {
    fprintf(stderr,"Error copying data to device.\n");
    failed[0]=1;
  }
  else
    failed[0]=0;

}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void free_ptr(long *ptr)
{
  void *tmp=(void *)(ptr[0]);
  if (cudaFree(tmp)!=cudaSuccess) {
    fprintf(stderr,"Failed cudaFree.\n");
  }
}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void sgemm(int transa, int transb, int n, int m, int k, long *A, long *B, long *C)
{
  //printf("transa and transb are %d %d\n",transa, transb);
  float *dA=(float *)(A[0]);
  float *dB=(float *)(B[0]);
  float *dC=(float *)(C[0]);
  cublasStatus_t stat;
  cublasHandle_t handle;
  
  stat=cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return;
  }
  float one=1.0;
  float zero=0.0;

  //printf("looking for match.\n");
  if ((transa==0)&&(transb==0)) {
    printf("calling standard sgemm.\n");
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,dB,m,dA,k,&zero,dC,m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS sgemm failed\n");
      return;      
    }
  }
  if ((transa==0)&&(transb)) {
    stat=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&one,dB,k,dA,k,&zero,dC,m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS sgemm failed\n");
      return;      
    }
  }
  if ((transa)&&(transb==0)) {
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,n,k,&one,dB,m,dA,n,&zero,dC,m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS sgemm failed\n");
      return;      
    }
  }
  if ((transa)&&(transb)) {
    stat=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,&one,dB,k,dA,n,&zero,dC,m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS sgemm failed\n");
      return;      
    }
  }
}
}
