//nvcc -o libgpu_point.so gpu_point.cu -shared -lcublas -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// get_time(), time_since()
#include <ksgpu/time_utils.hpp>


__global__
void  fillA(float *dra_bore, float *ddec_bore, int n,float *dA)
{
  //int ra_ord=4;
  //int dec_ord=3;
  //int t_ord=2;
  int myi_off=blockIdx.x*blockDim.x+threadIdx.x;
  int nthread=blockDim.x*gridDim.x;
  for (int myind=myi_off;myind<n;myind+=nthread) {
    float myra=dra_bore[myind];
    float mydec=ddec_bore[myind];
    float tmp=myra*myra;

    dA[myind]=1.0;
    float t=2.0*(myind-(n-1))/(n-1.0)+1.0;
    dA[myind+1*n]=t;
    dA[myind+2*n]=t*t;


    dA[myind+6*n]=myra;
    dA[myind+7*n]=tmp;
    dA[myind+8*n]=myra*tmp;
    dA[myind+9*n]=tmp*tmp;
    tmp=mydec*mydec;
    dA[myind+3*n]=mydec;
    dA[myind+4*n]=tmp;
    dA[myind+5*n]=tmp*mydec;
    dA[myind+10*n]=myra*t;
    dA[myind+11*n]=mydec*t;
    dA[myind+12*n]=mydec*myra;
  }
}
/*--------------------------------------------------------------------------------*/
void print_arr(float *dvec, int n)
{
  float *hvec=(float *)malloc(sizeof(float)*n);
  if (cudaMemcpy(hvec,dvec,n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess) {
    fprintf(stderr,"Error copying from device in print_arr.\n");
    return;
  }
  for (int i=0;i<n;i++)
    printf("%5d %14.6g\n",i,hvec[i]);
  free(hvec);
      

}
/*--------------------------------------------------------------------------------*/
extern "C" {
void fillA_host(long *dra_bore,long *ddec_bore,int n, long *dA)
{
  float *raptr=(float *)(dra_bore[0]);
  float *decptr=(float *)(ddec_bore[0]);
  float *dAptr=(float *)(dA[0]);
  
  fillA<<<128,128>>>(raptr,decptr,n,dAptr);
  return;
  //printf("dec boresight:\n");
  //print_arr(ddec_bore,5);
  //fillA<<<128,128>>>(dra_bore,ddec_bore,n,dA);
}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void eval_fit(float *out,float *fitp, int n, int ndet, float *ra_bore, float *dec_bore)
{
  float *dA;
  int npar=13; //because we're super fragile right now
  if (cudaMalloc((void **)&dA,sizeof(float)*n*npar)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  float *dra_bore;
  if (cudaMalloc((void **)&dra_bore,sizeof(float)*n*npar)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");

  float *ddec_bore;
  if (cudaMalloc((void **)&ddec_bore,sizeof(float)*n*npar)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");

  float *dfitp;
  if (cudaMalloc((void **)&dfitp,sizeof(float)*ndet*npar)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");  
  if (cudaMemcpy(dfitp,fitp,ndet*npar*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  if (cudaMemcpy(dra_bore,ra_bore,n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying ra bore data to device.\n");
  if (cudaMemcpy(ddec_bore,dec_bore,n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying dec bore data to device.\n");


  float *dout;
  if (cudaMalloc((void **)&dout,sizeof(float)*ndet*n)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");  

  cublasHandle_t handle;
  cublasStatus_t stat;
  stat=cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return;
  }
  float one=1.0;
  float zero=0.0;

  cudaDeviceSynchronize();
  for (int i=0;i<10;i++) {
      struct timeval tv = ksgpu::get_time();
    fillA<<<128,128>>>(dra_bore,ddec_bore,n,dA);
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,n,ndet,npar,&one,dA,n,dfitp,ndet,&zero,dout,n);
    if (stat!=CUBLAS_STATUS_SUCCESS) 
      printf("Error in sgemm.\n");
    cudaDeviceSynchronize();
    double dt = ksgpu::time_since(tv);
    printf("Pointing reconstruction took %12.4g\n",dt);
  }
  if (cudaMemcpy(out,dout,n*ndet*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying out back to host.\n");
  cudaFree(dA);
  cudaFree(dra_bore);
  cudaFree(ddec_bore);
  cudaFree(dfitp);
}
}
