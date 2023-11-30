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


static void test_adjointness(const ActPointing &ap)
{
    gputils::Array<float> m({3,ap.ndec,ap.nra}, af_rhost | af_random);
    gputils::Array<float> t({ap.ndet,ap.nt}, af_rhost | af_random);

    gputils::Array<float> At({3,ap.ndec,ap.nra}, af_rhost | af_zero);  // same shape as map
    reference_tod2map(At, t, ap.xpointing);
    
    gputils::Array<float> Am({ap.ndet,ap.nt}, af_rhost | af_zero);     // same shape as tod
    reference_map2tod(Am, m, ap.xpointing);

    float dot1 = vdot(m, At);
    float dot2 = vdot(Am, t);
    cout << "test_adjointness: " << dot1 << " " << dot2 << endl;
}


int main(int argc, char **argv)
{
    if (argc != 2) {
	cerr << "Usage: " << argv[0] << " /home/kmsmith/xpointing/xpointing_0.npz" << endl;
	exit(2);
    }
    
    ActPointing ap(argv[1]);

    for (int i = 0; i < 10; i++)
	test_adjointness(ap);

    // cout << "test-tod2map: pass" << endl;
    
    return 0;
}
