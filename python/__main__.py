import sys

if (len(sys.argv) == 2) and (sys.argv[1] == 'test'):
    from . import tests
    tests.test_pointing_preplan()
#elif (len(sys.argv) == 2) and (sys.argv[1] == 'time'):
#    from . import timing
#    timing.time_points2alm(npoints_per_gpu=1000*1000, lmax=1000)
else:
    print(f'Usage: python -m direct_sht [test | time]')
        
