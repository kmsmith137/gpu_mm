#include "../include/gpu_mm.hpp"

#include <iostream>
#include <algorithm>   // std::sort()

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


struct Quadruple
{
    // Sentinel values.
    int cell_idec = -1;   // divisible by 64 (except for sentinel)
    int cell_ira = -1;    // divisible by 64 (except for sentinel)
    int icl_start = 0;
    int icl_end = -1;
};


struct Triple
{
    int cell_idec;  // divisible by 64
    int cell_ira;   // divisible by 64
    int cltod;

    // For std::sort()
    bool operator<(const Triple &x) const
    {
	if (cell_idec < x.cell_idec)
	    return true;
	if (cell_idec > x.cell_idec)
	    return false;
	if (cell_ira < x.cell_ira)
	    return true;
	if (cell_ira > x.cell_ira)
	    return false;
	return (cltod < x.cltod);
    }

    // For _add_triple()
    bool operator==(const Triple &x) const
    {
	return (cell_idec == x.cell_idec) && (cell_ira == x.cell_ira) && (cltod == x.cltod);
    }
};


inline void _add_triple(vector<Triple> &v, size_t prev_size, int idec, int ira, int cltod)
{
    Triple t;
    t.cell_idec = idec & ~63;
    t.cell_ira = ira & ~63;
    t.cltod = cltod;

    for (size_t i = prev_size; i < v.size(); i++)
	if (v[i] == t)
	    return;

    v.push_back(t);
}



OldPointingPlan::OldPointingPlan(const Array<float> &xpointing, int ndec, int nra, bool verbose)
{
    long nsamp;
    check_xpointing_and_init_nsamp(xpointing, nsamp, "OldPointingPlan constructor", false);  // on_gpu=false
    check_nypix_global(ndec, "OldPointingPlanConstructor");
    check_nxpix_global(nra, "OldPointingPlanConstructor");

    const float *xp = xpointing.data;
    
    // Step 1: Construct and sort a vector<Triple>.
    
    vector<Triple> tvec;
    this->ncl_uninflated = nsamp / 32;
    
    for (long cltod = 0; cltod < ncl_uninflated; cltod++) {
	size_t prev_size = tvec.size();

	for (long s = 32*cltod; s < 32*(cltod+1); s++) {
	    float px_dec = xp[s];
	    float px_ra = xp[s+nsamp];
	    
	    int idec = int(px_dec);
	    int ira = int(px_ra);
	    float ddec = px_dec - float(idec);
	    float dra = px_ra - float(ira);
	
	    xassert(idec >= 0);
	    xassert(idec < ndec-1);
	    xassert(ira >= 0);
	    xassert(ira < nra-1);

	    _add_triple(tvec, prev_size, idec, ira, cltod);
	    _add_triple(tvec, prev_size, idec, ira+1, cltod);
	    _add_triple(tvec, prev_size, idec+1, ira, cltod);
	    _add_triple(tvec, prev_size, idec+1, ira+1, cltod);
	}
    }

    std::sort(tvec.begin(), tvec.end());
    this->ncl_inflated = tvec.size();

    // Step 2: Construct and sort a vector<Quadruple>.

    vector<Quadruple> qvec;
    qvec.push_back(Quadruple());  // qvec[0] is a sentinel
    
    for (long icl = 0; icl < ncl_inflated; icl++) {
	const Triple &t = tvec[icl];
	Quadruple &qprev = qvec[qvec.size()-1];
	
	if ((qprev.cell_idec == t.cell_idec) && (qprev.cell_ira == t.cell_ira))
	    continue;

	// Note: this line must precede qvec.push_back(...), to ensure 'qprev' pointer is still valid.
	qprev.icl_end = icl;
	
	Quadruple qnew;
	qnew.cell_idec = tvec[icl].cell_idec;
	qnew.cell_ira = tvec[icl].cell_ira;
	qnew.icl_start = icl;
	qnew.icl_end = -1;
	qvec.push_back(qnew);
    }

    qvec[qvec.size()-1].icl_end = ncl_inflated;
    
    // Step 3: let's check consistency of tvec + qvec.
    // Note that qvec[0] is a sentinel, and the "real" qvec starts at 1.

    long nq = qvec.size()-1;
    xassert(nq > 0);
    xassert(qvec[1].icl_start == 0);
    xassert(qvec[nq].icl_end == ncl_inflated);

    for (long q = 1; q < nq; q++)
	xassert(qvec[q].icl_end == qvec[q+1].icl_start);

    for (long q = 1; q <= nq; q++) {
	int icl0 = qvec[q].icl_start;
	int icl1 = qvec[q].icl_end;
	
	xassert(icl0 >= 0);
	xassert(icl0 < icl1);
	xassert(icl1 <= ncl_inflated);
	
	for (int icl = icl0; icl < icl1; icl++) {
	    xassert(tvec[icl].cell_idec == qvec[q].cell_idec);
	    xassert(tvec[icl].cell_ira == qvec[q].cell_ira);
	}
    }

    // Step 4: Now create the plan arrays.
    
    this->num_quadruples = nq;
    this->plan_cltod_list = Array<int> ({ncl_inflated}, af_rhost);
    this->plan_quadruples = Array<int> ({nq,4}, af_rhost);

    for (long icl = 0; icl < ncl_inflated; icl++)
	plan_cltod_list.data[icl] = tvec[icl].cltod;
    
    for (long q = 0; q < nq; q++) {
	// Note (q+1) on RHS here.
	plan_quadruples.data[4*q] = qvec[q+1].cell_idec;
	plan_quadruples.data[4*q+1] = qvec[q+1].cell_ira;
	plan_quadruples.data[4*q+2] = qvec[q+1].icl_start;
	plan_quadruples.data[4*q+3] = qvec[q+1].icl_end;
    }

    if (!verbose)
	return;
  /*
    cout << "OldPointingPlan:\n"
	 << "    ncl_uninflated = " << ncl_uninflated << "\n"
	 << "    ncl_inflated = " << ncl_inflated << "\n"
	 << "    num_quadruples = " << num_quadruples << " (i.e. nonempty cells)\n"
	 << "    inflation factor = " << (double(ncl_inflated) / double(ncl_uninflated)) << "\n"
	 << "    cache lines per nonempty cell = " << (double(ncl_inflated) / double(num_quadruples)) << endl;
	*/
}


}  // namespace gpu_mm
