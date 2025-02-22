import sys

if (len(sys.argv) == 2) and (sys.argv[1] == 'test'):
    from . import tests
    tests.test_plan_iterator()
    tests.test_expand_dynamic_map()
    for p in tests.PointingInstance.generate_test_instances():
        print(p.name)
        p.test_all()
    print("Note: MPI tests must be run separately with 'mpiexec -np XX python -m gpu_mm test_mpi")

elif (len(sys.argv) == 2) and (sys.argv[1] == 'test_mpi'):
    from . import tests_mpi
    tests_mpi.test_mpi_pixelization()

elif (len(sys.argv) == 2) and (sys.argv[1] == 'time'):
    from . import tests
    for p in tests.PointingInstance.generate_timing_instances():
        print(p.name)
        p.time_all()

else:
    print(f'Usage: python -m direct_sht [test | time]')
        
