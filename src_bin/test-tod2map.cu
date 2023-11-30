#include "../include/gpu_mm.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/test_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm;


static float vdot(const float *x, const float *y, long n)
{
    float ret = 0.0;
    for (long i = 0; i < n; i++)
	ret += x[i] * y[i];
    return ret;
}
    

static float vdot(const Array<float> &x, const Array<float> &y)
{
    assert(x.shape_equals(y));
    assert(x.is_fully_contiguous());
    assert(y.is_fully_contiguous());
    assert(x.size == y.size);   // paranoid

    return vdot(x.data, y.data, x.size);
}


void test_adjointness(const ActPointing &ap)
{
    Array<float> m({3,ap.ndec,ap.nra}, af_rhost | af_random);
    Array<float> t({ap.ndet,ap.nt}, af_rhost | af_random);

    Array<float> At({3,ap.ndec,ap.nra}, af_rhost | af_zero);  // same shape as map
    reference_tod2map(At, t, ap.xpointing);
    
    Array<float> Am({ap.ndet,ap.nt}, af_rhost | af_zero);     // same shape as tod
    reference_map2tod(Am, m, ap.xpointing);

    float dot1 = vdot(m, At);
    float dot2 = vdot(Am, t);
    cout << "test_adjointness: " << dot1 << " " << dot2 << endl;
}


void test_tod2map_plan(const ActPointing &ap, const CpuPointingPlan &pp)
{
    Array<float> t({ap.ndet,ap.nt}, af_rhost | af_random);
    Array<float> m1({3,ap.ndec,ap.nra}, af_rhost | af_random);
    Array<float> m2({3,ap.ndec,ap.nra}, af_rhost | af_random);

    reference_tod2map(m1, t, ap.xpointing);                                          // m1 = no plan
    reference_tod2map(m2, t, ap.xpointing, pp.plan_cltod_list, pp.plan_quadruples);  // m2 = with plan

    assert_arrays_equal(m1, m2, "tod2map_no_plan", "tod2map_with_plan", {"iqu","idec","ira"});
    cout << "test_tod2map_plan(): pass" << endl;
}


void test_tod2map_gpu(const ActPointing &ap, const CpuPointingPlan &pp)
{
    Array<float> xpointing_gpu = ap.xpointing.to_gpu();
    Array<int> plan_cltod_list_gpu = pp.plan_cltod_list.to_gpu();
    Array<int> plan_quadruples_gpu = pp.plan_quadruples.to_gpu();
    
    Array<float> tod_cpu({ap.ndet,ap.nt}, af_rhost | af_random);
    Array<float> tod_gpu = tod_cpu.to_gpu();
    
    // FIXME need af_zero here, since GPU kernel doesn't properly zero its output
    Array<float> map_cpu({3,ap.ndec,ap.nra}, af_rhost | af_zero);
    Array<float> map_gpu({3,ap.ndec,ap.nra}, af_gpu | af_zero);

    launch_tod2map(map_gpu, tod_gpu, xpointing_gpu, plan_cltod_list_gpu, plan_quadruples_gpu);
    reference_tod2map(map_cpu, tod_cpu, ap.xpointing, pp.plan_cltod_list, pp.plan_quadruples);
    
    assert_arrays_equal(map_cpu, map_gpu, "tod2map_cpu", "tod2map_gpu", {"iqu","idec","ira"});
    cout << "test_tod2map_gpu(): pass" << endl;
}


int main(int argc, char **argv)
{
    if (argc != 2) {
	cerr << "Usage: " << argv[0] << " /home/kmsmith/xpointing/xpointing_0.npz" << endl;
	exit(2);
    }
    
    ActPointing ap(argv[1]);
    CpuPointingPlan pp(ap.xpointing, ap.ndec, ap.nra);

    for (int i = 0; i < 10; i++) {
	test_tod2map_gpu(ap, pp);
	// test_tod2map_plan(ap, pp);
	// test_adjointness(ap);
    }

    return 0;
}
