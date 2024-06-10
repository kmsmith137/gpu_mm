#include "../include/gpu_mm2.hpp"

#include <vector>
#include <iostream>
#include <algorithm> // std::sort
#include <gputils/cuda_utils.hpp>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm2;


struct ReferencePlan
{
    const long nsamp;
    const long nypix;
    const long nxpix;

    vector<ulong> sorted_mt;  // 46 bits (not 64), fully sorted
    vector<uint> nmt_cumsum;  // length nblocks
    
    // add_cell() temp state
    int nmt_tmp = 0;
    int tmp_cells[128];

    
    void add_tmp_cell(int iypix, int ixpix)
    {
	assert((iypix >= 0) && (iypix < nypix));
	assert((ixpix >= 0) && (ixpix < nxpix));

	int ycell = iypix >> 6;
	int xcell = ixpix >> 6;
	int icell = (ycell << 10) | xcell;

	for (int i = 0; i < nmt_tmp; i++)
	    if (tmp_cells[i] == icell)
		return;

	assert(nmt_tmp < 128);
	tmp_cells[nmt_tmp++] = icell;
    }

    
    template<typename T>
    ReferencePlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu) :
	nsamp(pp.nsamp), nypix(pp.nypix), nxpix(pp.nxpix)
    {
	long xpointing_nsamp = 0;
	check_xpointing(xpointing_gpu, xpointing_nsamp, "ReferencePlan constructor");
	assert(xpointing_nsamp == pp.nsamp);

	QuantizedPointing qp(xpointing_gpu, nypix, nxpix);

	int rk = pp.rk;
	int nblocks = pp.nblocks;
	this->nmt_cumsum.resize(nblocks);
	
	// Initialize 'sorted_mt' and 'nmt_cumsum'.

	for (long b = 0; b < nblocks; b++) {
	    ulong s0 = min((b) << rk, nsamp);
	    ulong s1 = min((b+1) << rk, nsamp);
	    
	    while (s0 < s1) {
		this->nmt_tmp = 0;
		for (ulong s = s0; s < s0+32; s++) {
		    int iypix = qp.iypix_cpu.data[s];
		    int ixpix = qp.ixpix_cpu.data[s];
		    int ixpix1 = (ixpix < (nxpix-1)) ? (ixpix+1) : 0;
		    
		    this->add_tmp_cell(iypix, ixpix);
		    this->add_tmp_cell(iypix, ixpix1);
		    this->add_tmp_cell(iypix+1, ixpix);
		    this->add_tmp_cell(iypix+1, ixpix1);
		}

		std::sort(tmp_cells, tmp_cells + nmt_tmp);
		
		for (int i = 0; i < this->nmt_tmp; i++) {
		    ulong mt = ulong(tmp_cells[i]) | (ulong(s0 >> 5) << 20);
		    sorted_mt.push_back(mt);
		}
		    
		s0 += 32;
	    }
	    
	    nmt_cumsum[b] = sorted_mt.size();
	}
    }
};


template<typename T>
static void test_pointing_plan()
{
    long nsamp = 256*1024*1024;
    long nypix = 8*1024;
    long nxpix = 32*1024;
    
    double scan_speed = 0.531289;      // pixels per TOD sample
    double total_drift = 94289.38921;  // x-pixels

    ToyPointing<T> tp(nsamp, nypix, nxpix, scan_speed, total_drift);

    PointingPrePlan pp(tp.xpointing_gpu, nypix, nxpix);
    pp.show();

    Array<unsigned char> buf({pp.plan_nbytes}, af_gpu);
    Array<unsigned char> tmp_buf({pp.plan_constructor_tmp_nbytes}, af_gpu);
    PointingPlan p(pp, tp.xpointing_gpu, buf, tmp_buf);

    ReferencePlan rp(pp, tp.xpointing_gpu);

    // -------------------- Test preplan --------------------
    
    int rk = pp.rk;
    int nblocks = pp.nblocks;
    long nmt = pp.plan_nmt; 
    Array<uint> nmt_cumsum = pp.nmt_cumsum.to_host();

    // These asserts are kinda silly, but I wanted to put them somewhere.
    assert(nsamp > ((nblocks-1) << rk));
    assert(nsamp <= ((nblocks) << rk));
    assert(nmt_cumsum.data[nblocks-1] == nmt);
    
    for (int b = 0; b < nblocks; b++)
	assert(nmt_cumsum.data[b] == rp.nmt_cumsum[b]);

    // -------------------- Test that plan_mt is properly sorted --------------------
    
    Array<ulong> plan_mt = p.plan_mt_to_cpu();
    ulong *mt = plan_mt.data;

    ulong mask = (1L << 20) - 1;
    for (long i = 1; i < nmt; i++)
	assert((mt[i-1] & mask) <= (mt[i] & mask));

    // -------------------- Test plan_mt --------------------

    // First clear the 'sid' bits and re-sort.
    
    mask = (1L << 46) - 1;
    for (long i = 0; i < nmt; i++)
	mt[i] &= mask;

    std::sort(mt, mt + nmt);

    // Now compare the PointingPlan to the ReferencePlan.
    
    for (int i = 0; i < nmt; i++)
	assert(mt[i] == rp.sorted_mt[i]);
}


int main(int argc, char **argv)
{
    test_pointing_plan<float>();
    test_pointing_plan<double>();
    return 0;
}
