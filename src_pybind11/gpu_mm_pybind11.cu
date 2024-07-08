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


static const char *xstrdup(const string &s)
{
    const char *ret = strdup(s.c_str());
    return ret ? ret : "strdup() failed?";
}


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
    using PointingPlan = gpu_mm2::PointingPlan;

    string pp_suffix =
	"Reminder: plan creation is factored into two steps:\n"
	"\n"
	"   - Create PointingPrePlan from xpointing array\n"
	"   - Create PointingPlan from PointingPrePlan + xpointing array.\n"
	"\n"
	"These steps have similar running times, but the PointingPrePlan is much\n"
	"smaller in memory (a few KB versus ~100 MB). Therefore, PointingPrePlans\n"
	"can be retained (per-TOD) for the duration of the program, whereas\n"
	"PointingPlans will typically be created and destroyed on the fly.\n";
	
    // FIXME write longer docstrings
    string pointing_preplan_docstring =
	"PointingPrePlan(xpointing_gpu, nypix, nxpix)\n"
	"\n"
	+ pp_suffix;
    
    string pointing_plan_docstring =
	"PointingPlan(preplan, xpointing_gpu)\n"
	"\n"
	+ pp_suffix;


    // If updating this wrapper, don't forget to update comment in gpu_mm.py,
    // listing members/methods.
    py::class_<PointingPrePlan>(m, "PointingPrePlan", xstrdup(pointing_preplan_docstring))
	.def(py::init<const Array<Tmm>&, long, long>(),
	     py::arg("xpointing_gpu"), py::arg("nypix"), py::arg("nxpix"))
	
	.def_readonly("nsamp", &PointingPrePlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix", &PointingPrePlan::nypix, "Number of y-pixels")
	.def_readonly("nxpix", &PointingPrePlan::nxpix, "Number of x-pixels")
	.def_readonly("plan_nbytes", &PointingPrePlan::plan_nbytes, "Length of 'buf' argument to PointingPlan constructor")
	.def_readonly("plan_constructor_tmp_nbytes", &PointingPrePlan::plan_constructor_tmp_nbytes, "Length of 'tmp_buf' argument to PointingPlan constructor")
	
	.def_readonly("rk", &PointingPrePlan::rk, "Number of TOD samples per threadblock is (2^rk)")
	.def_readonly("nblocks", &PointingPrePlan::nblocks, "Number of threadblocks")
	.def_readonly("plan_nmt", &PointingPrePlan::plan_nmt, "Total number of mt-pairs in plan")
	.def_readonly("cub_nbytes", &PointingPrePlan::cub_nbytes, "Number of bytes used in cub radix sort 'd_temp_storage'")

	// FIXME temporary hack, used in tests.test_pointing_preplan().
	// To be replaced later by systematic API for shuffling between GPU/CPU.
	.def("get_nmt_cumsum", [](const PointingPrePlan &pp) { return pp.nmt_cumsum.to_host(); },
	     "Copies nmt_cumsum array to host, and returns it as a numpy array")
	
	.def("__str__", &PointingPrePlan::str)
    ;

        
    py::class_<PointingPlan>(m, "PointingPlan", xstrdup(pointing_plan_docstring))
	.def(py::init<const PointingPrePlan &, const Array<Tmm> &, const Array<unsigned char> &, const Array<unsigned char> &>(),
	     py::arg("preplan"), py::arg("xpointing_gpu"), py::arg("buf"), py::arg("tmp_buf"))
	
	.def_readonly("nsamp", &PointingPlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix", &PointingPlan::nypix, "Number of y-pixels")
	.def_readonly("nxpix", &PointingPlan::nxpix, "Number of x-pixels")

	// We wrap get_plan_mt() with the constraint on_gpu=false.
	// This is necessary because I wrote a to-python converter for numpy arrays, but not cupy arrays.
	// I might improve this later (not sure if it's necessary -- do we need on_gpu=true from python?)
	
	.def("get_plan_mt", [](const PointingPlan &p) { return p.get_plan_mt(false); },
	     "Length nmt_cumsum[-1] array, coarsely sorted by map cell")
	     
	.def("__str__", &PointingPlan::str)
    ;

    
    // ---------------------------------------------------------------------------------------------
    //
    // Only used in unit tests

    using ToyPointing = gpu_mm2::ToyPointing<Tmm>;
    using ReferencePointingPlan = gpu_mm2::ReferencePointingPlan;
    
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

    // FIXME write longer docstring
    const char *reference_pointing_plan_docstring =
	"ReferencePointingPlan: A utility class used in unit tests.\n";
        
    py::class_<ReferencePointingPlan>(m, "ReferencePointingPlan", reference_pointing_plan_docstring)
	.def(py::init<const PointingPrePlan &, const Array<Tmm> &>(),
	     py::arg("preplan"), py::arg("xpointing_gpu"))
	
	.def_readonly("nsamp", &ReferencePointingPlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix", &ReferencePointingPlan::nypix, "Number of y-pixels")
	.def_readonly("nxpix", &ReferencePointingPlan::nxpix, "Number of x-pixels")
	.def_readonly("rk", &ReferencePointingPlan::rk, "Number of TOD samples per threadblock is (2^rk)")
	.def_readonly("nblocks", &ReferencePointingPlan::nblocks, "Number of threadblocks")

	.def_readonly("iypix", &ReferencePointingPlan::iypix_arr, "Length-nsamp array containing integer y-pixel indices")
	.def_readonly("ixpix", &ReferencePointingPlan::ixpix_arr, "Length-nsamp array containing integer x-pixel indices")
	.def_readonly("nmt_cumsum", &ReferencePointingPlan::nmt_cumsum, "Length-nblocks array containing integer cumulative counts")
	.def_readonly("sorted_mt", &ReferencePointingPlan::sorted_mt, "Length nmt_cumsum[-1], see PointingPlan docstring for 'mt' format")
	     
	.def("__str__", &ReferencePointingPlan::str)
    ;
}
