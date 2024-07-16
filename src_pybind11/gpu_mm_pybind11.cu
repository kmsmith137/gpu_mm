// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in gputils/src_pybind11/gputils_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_gpu_mm

#include <iostream>
#include <gputils/pybind11.hpp>
#include "../include/gpu_mm.hpp"
#include "../include/plan_iterator.hpp"


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

    // Select template specialization T=Tmm
    auto _map2tod = [](const PointingPlan &self, Array<Tmm> &tod, const Array<Tmm> &lmap, const Array<Tmm> &xpointing, const LocalPixelization &lpix, bool allow_outlier_pixels, bool debug)
    {
	self.map2tod(tod, lmap, xpointing, lpix, allow_outlier_pixels, debug);
    };

    // Select template specialization T=Tmm
    auto _tod2map = [](const PointingPlan &self, Array<Tmm> &lmap, const Array<Tmm> &tod, const Array<Tmm> &xpointing, const LocalPixelization &lpix, bool allow_outlier_pixels, bool debug)
    {
	self.tod2map(lmap, tod, xpointing, lpix, allow_outlier_pixels, debug);
    };


    
    py::class_<LocalPixelization>(m, "LocalPixelization", "LocalPixelization: represents a set of map cells, held on a single GPU")
	.def(py::init<long, long, const Array<long>&, const Array<long>&, long, long, bool>(),
	     py::arg("nypix_global"), py::arg("nxpix_global"),
	     py::arg("cell_offsets_cpu"), py::arg("cell_offset_gpu"),
	     py::arg("ystride"), py::arg("polstride"), py::arg("periodic_xcoord"))
	;

    
    py::class_<PointingPrePlan>(m, "PointingPrePlan", xstrdup(pointing_preplan_docstring))
	.def(py::init<const Array<Tmm>&, long, long, Array<uint>&, Array<uint>&, bool, bool>(),
	     py::arg("xpointing_gpu"), py::arg("nypix_global"), py::arg("nxpix_global"),
	     py::arg("nmt_gpu"), py::arg("err_gpu"), py::arg("periodic_xcoord"), py::arg("debug"))

	.def_static("_get_preplan_size", []() { return PointingPrePlan::preplan_size; })

	.def_readonly("nsamp", &PointingPrePlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix_global", &PointingPrePlan::nypix_global, "Number of y-pixels")
	.def_readonly("nxpix_global", &PointingPrePlan::nxpix_global, "Number of x-pixels")
	
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

	.def("map2tod", _map2tod, py::arg("tod"), py::arg("local_map"), py::arg("xpointing"), py::arg("local_pixelization"), py::arg("allow_outlier_pixels") = false, py::arg("debug") = false)
	.def("tod2map", _tod2map, py::arg("local_map"), py::arg("tod"), py::arg("xpointing"), py::arg("local_pixelization"), py::arg("allow_outlier_pixels") = false, py::arg("debug") = false)

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

    using ToyPointing = gpu_mm::ToyPointing<Tmm>;
    using ReferencePointingPlan = gpu_mm::ReferencePointingPlan;
    using OldPointingPlan = gpu_mm::OldPointingPlan;

    // FIXME write longer docstring
    const char *reference_pointing_plan_docstring =
	"ReferencePointingPlan: A utility class used in unit tests.\n";

    
    // Select template specializations T=Tmm
    auto _reference_map2tod = [](Array<Tmm> &tod, const Array<Tmm> &lmap, const Array<Tmm> &xpointing, const LocalPixelization &lpix, bool periodic_xcoord, bool partial_pixelization)
    {
	gpu_mm::reference_map2tod(tod, lmap, xpointing, lpix, periodic_xcoord, partial_pixelization);
    };
    
    auto _reference_tod2map = [](Array<Tmm> &lmap, const Array<Tmm> &tod, const Array<Tmm> &xpointing, const LocalPixelization &lpix, bool periodic_xcoord, bool partial_pixelization)
    {
	gpu_mm::reference_tod2map(lmap, tod, xpointing, lpix, periodic_xcoord, partial_pixelization);
    };
	
    auto _unplanned_map2tod = [](Array<Tmm> &tod, const Array<Tmm> &lmap, const Array<Tmm> &xpointing, const LocalPixelization &lpix)
    {
	gpu_mm::launch_unplanned_map2tod(tod, lmap, xpointing, lpix);
    };
	
    auto _unplanned_tod2map = [](Array<Tmm> &lmap, const Array<Tmm> &tod, const Array<Tmm> &xpointing, const LocalPixelization &lpix)
    {
	gpu_mm::launch_unplanned_tod2map(lmap, tod, xpointing, lpix);
    };
    

    m.def("reference_map2tod", _reference_map2tod,
	  py::arg("tod"), py::arg("local_map"), py::arg("xpointing"), py::arg("local_pixelization"), py::arg("periodic_xcoord")=false, py::arg("partial_pixelization")=false);

    m.def("reference_tod2map", _reference_tod2map,
	  py::arg("local_map"), py::arg("tod"), py::arg("xpointing"), py::arg("local_pixelization"), py::arg("periodic_xcoord")=false, py::arg("partial_pixelization")=false);

    m.def("unplanned_map2tod", _unplanned_map2tod, "Warning: only implemented for allow_outlier_pixels=true!",
	  py::arg("tod"), py::arg("local_map"), py::arg("xpointing"), py::arg("local_pixelization"));

    m.def("unplanned_tod2map", _unplanned_tod2map, "Warning: only implemented for allow_outlier_pixels=true!",
	  py::arg("local_map"), py::arg("tod"), py::arg("xpointing"), py::arg("local_pixelization"));
	  
    m.def("old_map2tod", gpu_mm::launch_old_map2tod,
	  py::arg("tod"), py::arg("map"), py::arg("xpointing"));

    m.def("old_tod2map", gpu_mm::launch_old_tod2map,
	  py::arg("map"), py::arg("tod"), py::arg("xpointing"), py::arg("plan_cltod_list"), py::arg("plan_quadruples"));

    
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
        
    py::class_<ReferencePointingPlan>(m, "ReferencePointingPlan", reference_pointing_plan_docstring)
	.def(py::init<const PointingPrePlan &, const Array<Tmm> &, const Array<unsigned char> &>(),
	     py::arg("preplan"), py::arg("xpointing_gpu"), py::arg("tmp"))
	
	.def_readonly("nsamp", &ReferencePointingPlan::nsamp, "Number of TOD samples")
	.def_readonly("nypix_global", &ReferencePointingPlan::nypix_global, "Number of y-pixels")
	.def_readonly("nxpix_global", &ReferencePointingPlan::nxpix_global, "Number of x-pixels")
	
	.def_readonly("plan_nmt", &ReferencePointingPlan::plan_nmt, "Total number of mt-pairs in plan")
	.def_readonly("ncl_per_threadblock", &ReferencePointingPlan::ncl_per_threadblock, "Used when launching planner/preplanner kernels")
	.def_readonly("planner_nblocks", &ReferencePointingPlan::planner_nblocks, "Used when launching planner/preplanner kernels")

	.def_readonly("iypix", &ReferencePointingPlan::iypix_arr, "Length-nsamp array containing integer y-pixel indices")
	.def_readonly("ixpix", &ReferencePointingPlan::ixpix_arr, "Length-nsamp array containing integer x-pixel indices")
	.def_readonly("nmt_cumsum", &ReferencePointingPlan::nmt_cumsum, "Length-nblocks array containing integer cumulative counts")
	.def_readonly("sorted_mt", &ReferencePointingPlan::sorted_mt, "Length nmt_cumsum[-1], see PointingPlan docstring for 'mt' format")
	     
	.def_static("get_constructor_tmp_nbytes", &ReferencePointingPlan::get_constructor_tmp_nbytes, py::arg("preplan"))
		    
	.def("__str__", &ReferencePointingPlan::str)
    ;

    py::class_<OldPointingPlan>(m, "OldPointingPlan")
	.def(py::init<const Array<float> &, int, int, bool>(),
	     py::arg("xpointing"), py::arg("ndec"), py::arg("nra"), py::arg("verbose") = false)

	.def_readonly("_plan_cltod_list", &OldPointingPlan::plan_cltod_list)
	.def_readonly("_plan_quadruples", &OldPointingPlan::plan_quadruples)
    ;
					   
    m.def("test_plan_iterator", &gpu_mm::test_plan_iterator,
	  py::arg("plan_mt"), py::arg("nmt_per_block"), py::arg("warps_per_threadblock"));

    m.def("make_random_plan_mt", &gpu_mm::make_random_plan_mt,
	  py::arg("ncells"), py::arg("min_nmt_per_cell"), py::arg("max_nmt_per_cell"));
}
