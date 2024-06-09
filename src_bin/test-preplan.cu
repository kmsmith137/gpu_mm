#include "../include/gpu_mm2.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm2;


struct test_helper
{
    const int nypix;
    const int nxpix;

    int nmt = 0;
    int cells[128];

    test_helper(int nypix_, int nxpix_) :
	nypix(nypix_), nxpix(nxpix_) { }

    void add(int iypix, int ixpix)
    {
	assert((iypix >= 0) && (iypix < nypix));
	assert((ixpix >= 0) && (ixpix < nxpix));

	int ytile = iypix >> 6;
	int xtile = ixpix >> 6;
	int icell = (ytile << 10) | xtile;

	for (int i = 0; i < nmt; i++)
	    if (cells[i] == icell)
		return;

	assert(nmt < 128);
	cells[nmt++] = icell;
    }
};


template<typename T>
static void test_preplan()
{
    long nsamp = 256*1024*1024;
    long nypix = 8*1024;
    long nxpix = 32*1024;
    
    double scan_speed = 0.5;    // pixels per TOD sample
    double total_drift = 1024;  // x-pixels

    ToyPointing<T> tp(nsamp, nypix, nxpix, scan_speed, total_drift);

    PointingPrePlan pp(tp.xpointing_gpu, nypix, nxpix);
    Array<uint> nmt_cumsum = pp.nmt_cumsum.to_host();
    pp.show();
    
    QuantizedPointing qp(tp.xpointing_gpu, nypix, nxpix);

    // Accumulate nmt_cumsum by hand
    ulong s_curr = 0;
    ulong nmt_curr = 0;  // accumulated nmt[s] for s < s_curr
    test_helper h(nypix, nxpix);

    for (long b = 0; b < pp.nblocks; b++) {
	ulong s_end = min((b+1) << pp.rk, nsamp);

	while (s_curr < s_end) {
	    h.nmt = 0;
	    for (ulong s = s_curr; s < s_curr+32; s++) {
		int iypix = qp.iypix_cpu.data[s];
		int ixpix = qp.ixpix_cpu.data[s];
		int ixpix1 = (ixpix < (nxpix-1)) ? (ixpix+1) : 0;
		
		h.add(iypix, ixpix);
		h.add(iypix, ixpix1);
		h.add(iypix+1, ixpix);
		h.add(iypix+1, ixpix1);
	    }

	    nmt_curr += h.nmt;
	    s_curr += 32;
	}

	assert(s_curr == s_end);
	assert(nmt_cumsum.data[b] == nmt_curr);
    }

    assert(s_curr == nsamp);
    cout << "test_preplan< " << type_name<T>() << ">: pass" << endl;
}


int main(int argc, char **argv)
{
    test_preplan<float>();
    test_preplan<double>();
    return 0;
}
       
