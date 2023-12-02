#include "../include/gpu_mm.hpp"
#include <gputils/cuda_utils.hpp>  // CUDA_CALL


// Some global-variable hackery for the CpuPointingPlan constructor.
// I'd like to get rid of this eventually! (or at least make it thread-safe)
static std::shared_ptr<gpu_mm::CpuPointingPlan> plan_hack;
static long plan_hack_cookie = 4328943;


extern "C"
{
#if 0
}  // pacify emacs c-mode
#endif

void py_reference_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    gpu_mm::reference_map2tod(tod, map, xpointing, ndet, nt, ndec, nra);
}

void py_gpu_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    // FIXME here and elsewhere in this file, get rid of cudaDeviceSynchronize()
    CUDA_CALL(cudaDeviceSynchronize());
    gpu_mm::launch_map2tod(tod, map, xpointing, ndet, nt, ndec, nra);
    CUDA_CALL(cudaDeviceSynchronize());
}

void py_reference_tod2map(float *map, const float *tod, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    gpu_mm::reference_tod2map(map, tod, xpointing, ndet, nt, ndec, nra);
}

void py_gpu_tod2map(float *map, const float *tod, const float *xpointing, const int *plan_cltod_list,
		    const int *plan_quadruples, int plan_ncltod, int plan_nquadruples, int ndet, int nt,
		    int ndec, int nra)
{
    // FIXME here and elsewhere in this file, get rid of cudaDeviceSynchronize()
    CUDA_CALL(cudaDeviceSynchronize());
    gpu_mm::launch_tod2map(map, tod, xpointing, plan_cltod_list, plan_quadruples, plan_ncltod, plan_nquadruples, ndet, nt, ndec, nra);
    CUDA_CALL(cudaDeviceSynchronize());
}

// out[0] = cookie
// out[1] = ncl_uninflated
// out[2] = ncl_inflated
// out[3] = num_quadruples

void py_construct_cpu_plan1(long *out, const float *xpointing, int ndet, int nt, int ndec, int nra, int verbose)
{
    // Global variable hackery starts here...
    plan_hack_cookie++;
    plan_hack = std::make_shared<gpu_mm::CpuPointingPlan> (xpointing, ndet, nt, ndec, nra, verbose);

    out[0] = plan_hack_cookie;
    out[1] = plan_hack->ncl_uninflated;
    out[2] = plan_hack->ncl_inflated;
    out[3] = plan_hack->num_quadruples;
}

void py_construct_cpu_plan2(int *plan_cltod_list, int *plan_quadruples, long cookie, long ncl_inflated, long num_quadruples)
{
    std::shared_ptr<gpu_mm::CpuPointingPlan> p = plan_hack;
    plan_hack.reset();
    assert(p);
    
    assert(cookie == plan_hack_cookie);
    assert(ncl_inflated == p->ncl_inflated);
    assert(num_quadruples == p->num_quadruples);

    memcpy(plan_cltod_list, p->plan_cltod_list.data, ncl_inflated * sizeof(int));
    memcpy(plan_quadruples, p->plan_quadruples.data, 4 * num_quadruples * sizeof(int));
}


}  // extern "C"
