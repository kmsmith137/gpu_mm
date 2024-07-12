import sys

if (len(sys.argv) == 2) and (sys.argv[1] == 'test'):
    from . import tests
    tests.test_plan_iterator()
    for p in tests.PointingInstance.generate_test_instances():
        print(p.name)
        p.test_all()

elif (len(sys.argv) == 2) and (sys.argv[1] == 'time'):
    from . import tests
    for p in tests.PointingInstance.generate_timing_instances():
        print(p.name)
        p.time_all()

else:
    print(f'Usage: python -m direct_sht [test | time]')
        
