// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_gpu_mm

#include <iostream>
#include <ksgpu/pybind11.hpp>
#include "../include/gpu_mm.hpp"
#include "../include/plan_iterator.hpp"


using namespace std;
using namespace ksgpu;
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

    
    using LocalPixelization = gpu_mm::LocalPixelization;
    using PointingPrePlan = gpu_mm::PointingPrePlan;
    using PointingPlan = gpu_mm::PointingPlan;

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
	"PointingPrePlan(xpointing_gpu, nypix_global, nxpix_global)\n"
	"\n"
	+ pp_suffix;
    
    string pointing_plan_docstring =
	"PointingPlan(preplan, xpointing_gpu)\n"
	"\n"
	+ pp_suffix;


    
    py::class_<LocalPixelization>(m, "LocalPixelization", "LocalPixelization: represents a set of map cells, held on a single GPU")
	.def(py::init<long, long, const Array<long>&, const Array<long>&, long, long, bool>(),
	     py::arg("nypix_global"), py::arg("nxpix_global"),
	     py::arg("cell_offsets_cpu"), py::arg("cell_offset_gpu"),
	     py::arg("ystride"), py::arg("polstride"), py::arg("periodic_xcoord"))

	// Global pixelization
	.def_readonly("nypix_global", &LocalPixelization::nypix_global)
	.def_readonly("nxpix_global", &LocalPixelization::nxpix_global)
	.def_readonly("periodic_xcoord", &LocalPixelization::periodic_xcoord)

	// Local pixelization
	// Note: npix is mutable, since DynamicMap may change it.
	.def_readonly("ystride", &LocalPixelization::ystride)
	.def_readonly("polstride", &LocalPixelization::polstride)
	.def_readonly("nycells", &LocalPixelization::nycells)
	.def_readonly("nxcells", &LocalPixelization::nxcells)
	.def_readwrite("npix",  &LocalPixelization::npix, "counts only local pixels, does not include factor 3 from TQU.")

	// FIXME temporary kludge needed for DynamicMap, will go away later.
	.def("copy_gpu_offsets_to_cpu", &LocalPixelization::copy_gpu_offsets_to_cpu)
	.def("copy_cpu_offsets_to_gpu", &LocalPixelization::copy_cpu_offsets_to_gpu)
    ;

    
    py::class_<PointingPrePlan>(m, "PointingPrePlan", xstrdup(pointing_preplan_docstring))
	.def(py::init<const Array<Tmm>&, long, long, Array<uint>&, Array<uint>&, bool, bool>(),
	     py::arg("xpointing_gpu"), py::arg("nypix_global"), py::arg("nxpix_global"),
	     py::arg("nmt_gpu"), py::arg("err_gpu"), py::arg("periodic_xcoord"), py::arg("debug"))

	.def_static("_get_preplan_size", []() { return PointingPrePlan::preplan_size; })

	.def_readonly("nsamp", &PointingPrePlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix_global", &PointingPrePlan::nypix_global, "Number of y-pixels")
	.def_readonly("nxpix_global", &PointingPrePlan::nxpix_global, "Number of x-pixels")
	.def_readonly("periodic_xcoord", &PointingPrePlan::periodic_xcoord, "Boolean flag, indicates whether x-coordinate is periodic")
	
	.def_readonly("plan_nbytes", &PointingPrePlan::plan_nbytes, "Length of 'buf' argument to PointingPlan constructor")
	.def_readonly("plan_constructor_tmp_nbytes", &PointingPrePlan::plan_constructor_tmp_nbytes, "Length of 'tmp_buf' argument to PointingPlan constructor")
	.def_readonly("overhead", &PointingPrePlan::overhead, "typically ~0.3 (meaning that cell decomposition is a ~30% overhead)")
	
	.def_readonly("ncl_per_threadblock", &PointingPrePlan::ncl_per_threadblock, "Used when launching planner/preplanner kernels")
	.def_readonly("planner_nblocks", &PointingPrePlan::planner_nblocks, "Used when launching planner/preplanner kernels")
	
	.def_readonly("nmt_per_threadblock", &PointingPrePlan::nmt_per_threadblock, "Used in pointing operations (map2tod/tod2map)")
	.def_readonly("pointing_nblocks", &PointingPrePlan::pointing_nblocks, "Used in pointing operations (map2tod/tod2map)")
	
	.def_readonly("plan_nmt", &PointingPrePlan::plan_nmt, "Total number of mt-pairs in plan")
	.def_readonly("cub_nbytes", &PointingPrePlan::cub_nbytes, "Number of bytes used in cub radix sort 'd_temp_storage'")

	.def("get_nmt_cumsum", &PointingPrePlan::get_nmt_cumsum, "Copies nmt_cumsum array to host, and returns it as a numpy array.")
	
	.def("__str__", &PointingPrePlan::str)
    ;

    
    py::class_<PointingPlan>(m, "PointingPlan", xstrdup(pointing_plan_docstring))
	.def(py::init<const PointingPrePlan &, const Array<Tmm>&, const Array<unsigned char>&, const Array<unsigned char>&, bool>(),
	     py::arg("preplan"), py::arg("xpointing_gpu"), py::arg("buf"), py::arg("tmp_buf"), py::arg("debug")=false)
	
	.def_readonly("nsamp", &PointingPlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix_global", &PointingPlan::nypix_global, "Number of y-pixels")
	.def_readonly("nxpix_global", &PointingPlan::nxpix_global, "Number of x-pixels")
	.def_readonly("periodic_xcoord", &PointingPlan::periodic_xcoord, "Boolean flag, indicates whether x-coordinate is periodic")
	
	.def("_check_errflags", &PointingPlan::_check_errflags, py::arg("where"),
	     "I needed this once for tracking down a bug.")

	// We wrap get_plan_mt() with the constraint on_gpu=false.
	// This is necessary because I wrote a to-python converter for numpy arrays, but not cupy arrays.
	// I might improve this later (not sure if it's necessary -- do we need on_gpu=true from python?)
	
	.def("get_plan_mt", [](const PointingPlan &p) { return p.get_plan_mt(false); },
	     "Length nmt_cumsum[-1] array, coarsely sorted by map cell")

	.def("__str__", &PointingPlan::str)
    ;


    m.def("planned_map2tod", &gpu_mm::launch_planned_map2tod<Tmm>,
	  py::arg("tod"), py::arg("local_map"), py::arg("xpointing"),
	  py::arg("local_pixelization"), py::arg("plan"),
	  py::arg("partial_pixelization"), py::arg("debug"));    

    m.def("planned_tod2map", &gpu_mm::launch_planned_tod2map<Tmm>,
	  py::arg("local_map"), py::arg("tod"), py::arg("xpointing"),
	  py::arg("local_pixelization"), py::arg("plan"),
	  py::arg("partial_pixelization"), py::arg("debug"));    

    m.def("unplanned_map2tod", &gpu_mm::launch_unplanned_map2tod<Tmm>,
	  py::arg("tod"), py::arg("local_map"), py::arg("xpointing"),
	  py::arg("local_pixelization"), py::arg("errflags"),
	  py::arg("partial_pixelization"));

    m.def("unplanned_tod2map", &gpu_mm::launch_unplanned_tod2map<Tmm>,
	  py::arg("local_map"), py::arg("tod"), py::arg("xpointing"),
	  py::arg("local_pixelization"), py::arg("errflags"),
	  py::arg("partial_pixelization"));
    
    m.def("reference_map2tod", &gpu_mm::reference_map2tod<Tmm>,
	  py::arg("tod"), py::arg("local_map"), py::arg("xpointing"),
	  py::arg("local_pixelization"), py::arg("partial_pixelization"));

    m.def("reference_tod2map", &gpu_mm::reference_tod2map<Tmm>,
	  py::arg("local_map"), py::arg("tod"), py::arg("xpointing"),
	  py::arg("local_pixelization"), py::arg("partial_pixelization"));

    m.def("cell_broadcast", &gpu_mm::cell_broadcast<Tmm>,
	  py::arg("dst"), py::arg("src"), py::arg("index_map"));

    m.def("cell_reduce", &gpu_mm::cell_reduce<Tmm>,
	  py::arg("dst"), py::arg("src"), py::arg("index_map"));
    
    m.def("expand_dynamic_map", &gpu_mm::expand_dynamic_map,
	  py::arg("global_ncells"), py::arg("cell_offsets"), py::arg("plan_mt"));

    // FIXME temporary kludge that will go away later.
    m.def("expand_dynamic_map2", &gpu_mm::expand_dynamic_map2,
	  py::arg("global_ncells"), py::arg("local_pixelization"), py::arg("plan"));

    m.def("local_map_to_global", &gpu_mm::local_map_to_global<Tmm>,
	  py::arg("local_pixelization"), py::arg("dst"), py::arg("src"));
    
    // ---------------------------------------------------------------------------------------------
    //
    // Only used in unit tests

    using ToyPointing = gpu_mm::ToyPointing<Tmm>;
    using PointingPlanTester = gpu_mm::PointingPlanTester;
    
    
    py::class_<ToyPointing>(m, "ToyPointing")
	.def(py::init<long, long, double, double, const Array<Tmm>&, const Array<Tmm>&, bool>(),
	     py::arg("nypix_global"), py::arg("nxpix_global"),
	     py::arg("scan_speed"), py::arg("total_drift"),
	     py::arg("xpointing_cpu"), py::arg("xpointing_gpu"),
	     py::arg("noisy"))

	// .def_readonly("nsamp", &ToyPointing::nsamp, "Number of TOD samples")
	.def_readonly("nypix_global", &ToyPointing::nypix_global, "Number of y-pixels")
	.def_readonly("nxpix_global", &ToyPointing::nxpix_global, "Number of x-pixels")
	.def_readonly("scan_speed", &ToyPointing::scan_speed, "Scan speed in map pixels per TOD sample")
	.def_readonly("total_drift", &ToyPointing::total_drift, "Total drift over full TOD, in x-pixels")
	.def_readonly("drift_speed", &ToyPointing::drift_speed, "Drift (in x-pixels) per TOD sample")

	.def("__str__", &ToyPointing::str)
    ;
        
    py::class_<PointingPlanTester>(m, "PointingPlanTester", "PointingPlanTester: A utility class used in unit tests")
	.def(py::init<const PointingPrePlan &, const Array<Tmm> &, const Array<unsigned char> &>(),
	     py::arg("preplan"), py::arg("xpointing_gpu"), py::arg("tmp"))
	
	.def_readonly("nsamp", &PointingPlanTester::nsamp, "Number of TOD samples")
	.def_readonly("nypix_global", &PointingPlanTester::nypix_global, "Number of y-pixels")
	.def_readonly("nxpix_global", &PointingPlanTester::nxpix_global, "Number of x-pixels")
	
	.def_readonly("plan_nmt", &PointingPlanTester::plan_nmt, "Total number of mt-pairs in plan")
	.def_readonly("ncl_per_threadblock", &PointingPlanTester::ncl_per_threadblock, "Used when launching planner/preplanner kernels")
	.def_readonly("planner_nblocks", &PointingPlanTester::planner_nblocks, "Used when launching planner/preplanner kernels")

	.def_readonly("iypix", &PointingPlanTester::iypix_arr, "Length-nsamp array containing integer y-pixel indices")
	.def_readonly("ixpix", &PointingPlanTester::ixpix_arr, "Length-nsamp array containing integer x-pixel indices")
	.def_readonly("nmt_cumsum", &PointingPlanTester::nmt_cumsum, "Length-nblocks array containing integer cumulative counts")
	.def_readonly("sorted_mt", &PointingPlanTester::sorted_mt, "Length nmt_cumsum[-1], see PointingPlan docstring for 'mt' format")
	     
	.def_static("get_constructor_tmp_nbytes", &PointingPlanTester::get_constructor_tmp_nbytes, py::arg("preplan"))
		    
	.def("__str__", &PointingPlanTester::str)
    ;
					   
    m.def("test_plan_iterator", &gpu_mm::test_plan_iterator,
	  py::arg("plan_mt"), py::arg("nmt_per_block"), py::arg("warps_per_threadblock"));

    m.def("make_random_plan_mt", &gpu_mm::make_random_plan_mt,
	  py::arg("ncells"), py::arg("min_nmt_per_cell"), py::arg("max_nmt_per_cell"));
}
