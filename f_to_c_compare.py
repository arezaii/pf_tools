import numpy as np
import time
import pf_pytools.pf_fort_io as pf_fort_io
import pfio

input_pfb = '/home/arezaii/data/CONUS.5layer.pfclm.run4.out.clm_output.00002.C.pfb'

t3 = time.perf_counter()
fort_pfb_data = np.zeros((3342, 1888, 17), order='F')
pf_fort_io.pfb_read(fort_pfb_data, input_pfb)
fort_pfb_data = np.transpose(fort_pfb_data, (2, 1, 0))
t4 = time.perf_counter()
print(f'F load time {t4-t3:0.4f} seconds')

t1 = time.perf_counter()
c_pfb_data = pfio.pfread(input_pfb)
t2 = time.perf_counter()
print(f'C load time {t2-t1:0.4f} seconds')

if np.testing.assert_array_equal(fort_pfb_data, c_pfb_data) is None:
    print('arrays are equal')
