#include "../include/gpu_mm2.hpp"

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// FIXME all functions in this file are placeholders for future expansion.

void check_nsamp(long nsamp, const char *where)
{
    assert(nsamp > 0);
    assert(nsamp <= (1L << 31));
    assert((nsamp % 32) == 0);
}


void check_nypix_nxpix(long nypix, long nxpix, const char *where)
{
    assert(nypix > 0);
    assert(nypix <= 64*1024);
    assert((nypix % 64) == 0);
    
    assert(nxpix > 0);
    assert(nxpix <= 64*1024);
    assert((nxpix % 64) == 0);

    // See gpu_mm2.hpp for explanation of this constraint.
    assert((nypix <= 63*1024) || (nxpix <= 63*1024));
}


}  // namespace gpu_mm2
