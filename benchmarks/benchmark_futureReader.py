from usrpRoutines import *
import sys
from timingRoutines import Timer
import time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Inputs are (folder) (numSampsPerFile)")

    timer = Timer()

    folder = sys.argv[1]
    numSamps = int(sys.argv[2])

    #%%
    reader = SortedFolderReader(folder, numSamps, ensure_incremental=False)
    timer.start()
    reader.get(1, 1)
    timer.evt("first file read")

    time.sleep(3)
    timer.evt("sleeping")

    reader.get(1, 0)
    timer.evt("2nd file read, should be prefetched")

    reader.get(1, 2)
    timer.evt("3rd file read, manual again")

    time.sleep(3)
    timer.evt("sleeping")

    reader.get(1)
    timer.evt("4th file read, prefetched but not all consumed")

    reader.get(1, 1)
    timer.evt("5th file read, now prefetch is consumed")

    time.sleep(3)
    timer.evt("sleeping")

    reader.get(3)
    timer.evt("6th to 8th read, prefetch insufficient")
    
    timer.end()

    ####### Sample output
    # 0->1 : 2.361023s. first file read
    # 1->2 : 3.000033s. sleeping
    # 2->3 : 0.549124s. 2nd file read, should be prefetched
    # 3->4 : 2.366298s. 3rd file read, manual again
    # 4->5 : 3.000078s. sleeping
    # 5->6 : 0.551815s. 4th file read, prefetched but not all consumed
    # 6->7 : 0.651935s. 5th file read, now prefetch is consumed
    # 7->8 : 3.000037s. sleeping
    # 8->9 : 3.174942s. 6th to 8th read, prefetch insufficient
    # 9->10 : 0.000003s.
    # Total: 18.655288s.
