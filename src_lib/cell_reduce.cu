#include <iostream>
#include "../include/gpu_mm.hpp"

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif

template<typename T>
inline bool is_simple_1d(const Array<T> &a)
{
    return (a.ndim == 1) && (a.strides[0] == 1) && (a.on_host());
}

template<typename T>
void cell_reduce(Array<T> &dst, const Array<T> &src, const Array<long> &index_map)
{
    xassert(is_simple_1d(dst));
    xassert(is_simple_1d(src));
    xassert(is_simple_1d(index_map));

    int n = 3*64*64;
    long ndst = dst.shape[0];
    long ncells = index_map.shape[0];
    xassert(src.shape[0] == ncells*n);

    memset(dst.data, 0, ndst * sizeof(T));
    
    for (long i = 0; i < ncells; i++) {
	long idst = index_map.data[i] * n;   // note multiplication by n here
	xassert((idst >= 0) && (idst+n <= ndst));

	T *pdst = dst.data + idst;
	const T *psrc = src.data + i*n;

	for (int j = 0; j < n; j++)
	    pdst[j] += psrc[j];
    }
}


#define INSTANTIATE(T) \
    template void cell_reduce(Array<T> &, const Array<T> &, const Array<long> &)

INSTANTIATE(float);
INSTANTIATE(double);

}  // namespace gpu_mm
