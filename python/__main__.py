import sys

if (len(sys.argv) == 2) and (sys.argv[1] == 'test'):
    from . import tests
    for p in tests.PointingInstance.generate_test_instances():
        print(p.name)
        p.test_all()

elif (len(sys.argv) == 2) and (sys.argv[1] == 'time'):
    from . import tests
    for p in tests.PointingInstance.generate_timing_instances():
        print(p.name)
        p.time_pointing_preplan()
        p.time_pointing_plan()

else:
    print(f'Usage: python -m direct_sht [test | time]')
        
