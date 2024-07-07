// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in gputils/src_pybind11/gputils_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_gpu_mm

#include <iostream>
#include <gputils/pybind11.hpp>
#include "../include/gpu_mm2.hpp"


using namespace std;
using namespace gputils;
namespace py = pybind11;


// Currently we wrap either float32 or float64, determined at compile time!
// FIXME eventually, should wrap both.
using Tmm = float;


PYBIND11_MODULE(gpu_mm_pybind11, m)  // extension module gets compiled to gpu_mm_pybind11.so
{
    m.doc() = "gpu_mm: low-level library for CMB map-making on GPUs";

    // Note: looks like _import_array() will fail if different numpy versions are
    // found at compile-time versus runtime.

    if (_import_array() < 0) {
	PyErr_Print();
	PyErr_SetString(PyExc_ImportError, "gpu_mm: numpy.core.multiarray failed to import");
	return;
    }

    // FIXME temporary hack
    m.def("_get_tsize", []() { return sizeof(Tmm); });

    // ---------------------------------------------------------------------------------------------

    using PointingPrePlan = gpu_mm2::PointingPrePlan;

    const char *pointing_preplan_docstring =
	"PointingPrePlan(xpointing_gpu, nypix, nxpix)\n"
	"\n"
	"We factor plan creation into two steps:\n"
	"\n"
	"   - Create PointingPrePlan from xpointing array\n"
	"   - Create PointingPlan from PointingPrePlan + xpointing array.\n"
	"\n"
	"These steps have similar running times, but the PointingPrePlan is much\n"
	"smaller in memory (a few KB versus ~100 MB). Therefore, PointingPrePlans\n"
	"can be retained (per-TOD) for the duration of the program, whereas\n"
	"PointingPlans will typically be created and destroyed on the fly.\n";

    py::class_<PointingPrePlan>(m, "PointingPrePlan", pointing_preplan_docstring)
	.def(py::init<const Array<Tmm>&, long, long>(),
	     py::arg("xpointing_gpu"), py::arg("nypix"), py::arg("nxpix"))
	
	.def_readonly("nsamp", &PointingPrePlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix", &PointingPrePlan::nypix, "Number of y-pixels")
	.def_readonly("nxpix", &PointingPrePlan::nxpix, "Number of x-pixels")
	.def_readonly("plan_nbytes", &PointingPrePlan::plan_nbytes, "Length of 'buf' argument to PointingPlan constructor")
	.def_readonly("plan_constructor_tmp_nbytes", &PointingPrePlan::plan_constructor_tmp_nbytes, "Length of 'tmp_buf' argument to PointingPlan constructor")
	
	.def_readonly("_rk", &PointingPrePlan::rk, "Number of TOD samples per threadblock is (2^rk)")
	.def_readonly("_nblocks", &PointingPrePlan::nblocks, "Number of threadblocks")
	.def_readonly("_plan_nmt", &PointingPrePlan::plan_nmt, "Total number of mt-pairs in plan")
	.def_readonly("_cub_nbytes", &PointingPrePlan::cub_nbytes, "Number of bytes used in cub radix sort 'd_temp_storage'")

	// FIXME temporary hack, used in tests.test_pointing_preplan().
	// To be replaced later by systematic API for shuffling between GPU/CPU.
	.def("get_nmt_cumsum", [](const PointingPrePlan &pp) { return pp.nmt_cumsum.to_host(); },
	     "Copies nmt_cumsum array to host, and returns it as a numpy array")
	
	.def("__str__", &PointingPrePlan::str)
    ;

    // ---------------------------------------------------------------------------------------------
    //
    // Only used in unit tests

    using ToyPointing = gpu_mm2::ToyPointing<Tmm>;
    using QuantizedPointing = gpu_mm2::QuantizedPointing;
    
    py::class_<ToyPointing>(m, "ToyPointing")
	.def(py::init<long, long, long, double, double, const Array<Tmm>&, const Array<Tmm>&, bool>(),
	     py::arg("nsamp"), py::arg("nypix"), py::arg("nxpix"),
	     py::arg("scan_speed"), py::arg("total_drift"),
	     py::arg("xpointing_cpu"), py::arg("xpointing_gpu"),
	     py::arg("noisy"))

	.def_readonly("nsamp", &ToyPointing::nsamp, "Number of TOD samples")
	.def_readonly("nypix", &ToyPointing::nypix, "Number of y-pixels")
	.def_readonly("nxpix", &ToyPointing::nxpix, "Number of x-pixels")
	.def_readonly("scan_speed", &ToyPointing::scan_speed, "Scan speed in map pixels per TOD sample")
	.def_readonly("total_drift", &ToyPointing::total_drift, "Total drift over full TOD, in x-pixels")
	.def_readonly("drift_speed", &ToyPointing::drift_speed, "Drift (in x-pixels) per TOD sample")

	.def("__str__", &ToyPointing::str)
    ;

    const char *quantized_pointing_docstring =
	"QuantizedPointing: A utility class used in unit tests.\n"
	"\n"
	"Given an 'xpointing' array on the GPU, determine which map pixel (iy, ix)\n"
	"each time sample falls into, and store the result in two length-nsamp arrays\n"
	"(iypix_cpu, ixpix_cpu). (Since this class is only used in unit tests, we assume\n"
	"that the caller wants these arrays on the CPU.)\n";
        
    py::class_<QuantizedPointing>(m, "QuantizedPointing", quantized_pointing_docstring)
	.def(py::init<const Array<Tmm>&, long, long>(),
	     py::arg("xpointing_gpu"), py::arg("nypix"), py::arg("nxpix"))
	
	.def_readonly("nsamp", &QuantizedPointing::nsamp, "Number of TOD samples")
	.def_readonly("nypix", &QuantizedPointing::nypix, "Number of y-pixels")
	.def_readonly("nxpix", &QuantizedPointing::nxpix, "Number of x-pixels")

	.def_readonly("iypix_cpu", &QuantizedPointing::iypix_cpu, "Length-nsamp array containing integer y-pixel indices")
	.def_readonly("ixpix_cpu", &QuantizedPointing::ixpix_cpu, "Length-nsamp array containing integer x-pixel indices")

	.def("compute_nmt_cumsum", &QuantizedPointing::compute_nmt_cumsum, py::arg("rk"),
	     "For testing PointingPrePlan.\n"
	     "The 'rk' argument has the same meaning as PointingPrePlan.rk.\n"
	     "The returned array has the same meaning as PointingPrePlan.nmt_cumsum.")
	     
	.def("__str__", &QuantizedPointing::str)
    ;
}
