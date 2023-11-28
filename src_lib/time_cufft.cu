//nvcc -o libtime_cufft.so time_cufft.cu -shared -lcufft -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <omp.h>
#include <cuComplex.h>





void cufft_c2r(float *out, cufftComplex *data, int len, int ntrans, int isodd)
{
  int nout=2*(len-1)-isodd;
  //float *out;
  //cudaMalloc(&out,sizeof(float)*nout*ntrans);
  cufftHandle plan;
  
  if (cufftPlan1d(&plan,nout,CUFFT_C2R, ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning dft\n");
  for (int i=0;i<20;i++) {
    cudaDeviceSynchronize();
    double t1=omp_get_wtime();
    if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
      fprintf(stderr,"Error executing dft\n");
    cudaDeviceSynchronize();
    double t2=omp_get_wtime();
    printf("took %12.4g seconds to do fft.\n",t2-t1);
  }
  if (cufftDestroy(plan)!= CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
}

/*--------------------------------------------------------------------------------*/
void cufft_c2r_columns(float *out, cufftComplex *data,int len, int ntrans, int isodd)
{
  int nout=2*(len-1)+isodd;
  cufftHandle plan;
  int rank=1;
  int inembed[rank] = {ntrans};
  int onembed[rank]={ntrans};
  int istride=ntrans;
  int idist=1;
  int ostride=ntrans;
  int odist=1;
  if (cufftPlanMany(&plan,rank,&nout,inembed,istride,idist,onembed,ostride,odist,CUFFT_C2R,ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning DFT in c2r_columns.\n");
  if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing DFT in c2r_columns.\n");
  if (cufftDestroy(plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan in c2r_columns.\n");

}

/*--------------------------------------------------------------------------------*/
extern "C" {
void cufft_c2r_host(float *out, cufftComplex *data, int n, int m, int isodd,int axis)
{
  float *dout;
  cufftComplex *din;
  int nn;
  if (axis==0)
    nn=2*(n-1)+isodd;
  else
    nn=2*(m-1)+isodd;
  if (cudaMalloc((void **)&din,sizeof(cufftComplex)*n*m)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(din,data,n*m*sizeof(cufftComplex),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  if (axis==0) {
    if (cudaMalloc((void **)&dout,sizeof(float)*nn*m)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_c2r_columns(dout,din,n,m,isodd);
    //printf("copying %d %d\n",nn,m);
    if (cudaMemcpy(out,dout,sizeof(float)*nn*m,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in c2r\n");
  }
  else {
    if (cudaMalloc((void **)&dout,sizeof(float)*n*nn)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_c2r(dout,din,m,n,isodd);
    //printf("copying %d %d\n",n,nn);
    if (cudaMemcpy(out,dout,sizeof(float)*nn*n,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in c2r\n");
  

  }
}
}

/*--------------------------------------------------------------------------------*/
void cufft_r2c(cufftComplex *out, float *data, int len, int ntrans)
{
  //int nout=len/2+1;
  cufftHandle plan;
  
  if (cufftPlan1d(&plan,len,CUFFT_R2C, ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning dft\n");
  for (int i=0;i<20;i++)
    {
      cudaDeviceSynchronize();
      double t1=omp_get_wtime();
      if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
	fprintf(stderr,"Error executing dft\n");
      cudaDeviceSynchronize();
      double t2=omp_get_wtime();
      printf("r2c took %12.4g\n",t2-t1);
    }
  if (cufftDestroy(plan)!= CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
}

/*--------------------------------------------------------------------------------*/
void cufft_r2c_columns(cufftComplex *out, float *data, int len, int ntrans)
{
  //int nout=len/2+1;
  //printf("performing %d transforms of length %d %d\n",ntrans,len,nout);

  cufftHandle plan;
  int rank=1;
  int inembed[rank] = {len};
  int onembed[rank]={ntrans};
  int istride=ntrans;
  int idist=1;
  int ostride=ntrans;
  int odist=1;
  //if (cufftPlanMany(&plan,1,&nout,&one,len,1,&one,nout,1,CUFFT_R2C,ntrans)!=CUFFT_SUCCESS)
  //if (cufftPlanMany(&plan,rank,&len,inembed,len,1,onembed,nout,1,CUFFT_R2C,ntrans)!=CUFFT_SUCCESS)
  if (cufftPlanMany(&plan,rank,&len,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning DFT in r2c_columns.\n");
  if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing DFT in r2c_columns.\n");
  if (cufftDestroy(plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan in r2c_columns.\n");
  
}


/*--------------------------------------------------------------------------------*/

extern "C" {
void cufft_r2c_host(cufftComplex *out, float *data, int n, int m, int axis)
{
  cufftComplex *dout;
  float *din;
  int nn;
  if (axis==0)
    nn=n/2+1;
  else
    nn=m/2+1;
  if (cudaMalloc((void **)&din,sizeof(float)*n*m)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(din,data,n*m*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  if (axis==0) {
    if (cudaMalloc((void **)&dout,sizeof(cufftComplex)*nn*m)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_r2c_columns(dout,din,n,m);
    //printf("copying %d %d\n",nn,m);
    if (cudaMemcpy(out,dout,sizeof(cufftComplex)*nn*m,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in r2c\n");
  }
  else {
    if (cudaMalloc((void **)&dout,sizeof(cufftComplex)*n*nn)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_r2c(dout,din,m,n);
    //printf("copying %d %d\n",n,nn);
    if (cudaMemcpy(out,dout,sizeof(cufftComplex)*nn*n,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in r2c\n");
  
  }
}
}



/*================================================================================*/


#if 0

int main(int argc, char *argv[])
{
  printf("Hello world!\n");
  int ndet=1000;
  int nsamp=1<<18;
  printf("nsamp is %d\n",nsamp);

  float *fdat=(float *)malloc(sizeof(float)*ndet*nsamp);
  if (fdat!=NULL)
    printf("successfully malloced array on host.\n");

  float *ddat;
  if (cudaMalloc((void **)&ddat,sizeof(float)*nsamp*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  cuComplex *dtrans;
  if (cudaMalloc((void **)&dtrans,sizeof(cuComplex)*nsamp*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");

  
  
}
#endif
