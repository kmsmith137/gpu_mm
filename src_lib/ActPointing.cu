#include "../include/gpu_mm.hpp"
#include "../include/cnpy.hpp"

#include <cassert>
#include <iostream>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


static long round_up(long n, long m)
{
    assert(m > 0);
    assert(n >= 0);
    return m * ((n+m-1) / m);
}


// FIXME I'd like to generalize this to include 
struct TodStats
{
    float minval;
    float maxval;

    TodStats(const float *p, long n)
    {
	assert(n > 0);
	minval = maxval = p[0];
	
	for (long i = 1; i < n; i++) {
	    minval = std::min(minval, p[i]);
	    maxval = std::max(maxval, p[i]);
	}
    }
};



ActPointing::ActPointing(const string &filename) :
    xpointing_npz_filename(filename)
{
    cout << "Reading " << filename << endl;
    cnpy::npz_t npz = cnpy::npz_load(filename);

    auto p = npz.find("xpointing");
    if (p == npz.end())
	throw runtime_error(filename + ": key 'xpointing' not found");

    const cnpy::NpyArray &xp = (*p).second;
    cout << filename << ": xpointing.shape=" << gputils::tuple_str(xp.shape) << endl;

    assert(xp.shape.size() == 3);
    assert(xp.shape[0] == 3);
    assert(xp.word_size == sizeof(float));
    assert(!xp.fortran_order);

    // Convert vector<size_t> -> vector<ssize_t>
    int ndim = xp.shape.size();
    vector<ssize_t> shape(ndim);
    for (int i = 0; i < ndim; i++)
	shape[i] = xp.shape[i];
    
    this->xpointing = gputils::Array<float> (shape, af_rhost);
    this->ndet = xp.shape[1];
    this->nt = xp.shape[2];
        
    long ns = long(ndet) * long(nt);
    assert(xp.num_bytes() == 3 * ns * sizeof(float));
    
    // Note: in pointing files, the length-3 axis is ordered { ra, dec, alpha }.
    // Internally in C++/cuda code, we usually re-order to { dec, ra, alpha }.

    memcpy(this->xpointing.data, xp.data<float>() + ns, ns * sizeof(float));   // dec (index 1->0)
    memcpy(this->xpointing.data + ns, xp.data<float>(), ns * sizeof(float));   // ra (index 0->1)
    memcpy(this->xpointing.data + 2*ns, xp.data<float>() + 2*ns, ns * sizeof(float));   // alpha (index 2->2)
    
    TodStats dec_stats(xpointing.data, ns);
    TodStats ra_stats(xpointing.data + ns, ns);

    cout << filename << ": px_dec min=" << dec_stats.minval << ", max=" << dec_stats.maxval << endl;
    cout << filename << ": px_ra min=" << ra_stats.minval << ", max=" << ra_stats.maxval << endl;
    
    assert(dec_stats.minval > 0.01);
    assert(ra_stats.minval > 0.01);
    assert(dec_stats.maxval < 100000);
    assert(ra_stats.maxval < 100000);

    this->ndec = round_up(dec_stats.maxval + 2, 64);
    this->nra = round_up(ra_stats.maxval + 2, 64);
    cout << "filename: ndet=" << ndet << ", nt=" << nt << ", ndec=" << ndec << ", nra=" << nra << endl;
}


}  // namespace gpu_mm
