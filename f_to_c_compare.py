import numpy as np
import time
import pf_pytools.pf_fort_io as pf_fort_io
import pfio

input_pfb = '/home/arezaii/Downloads/CONUS.5layer.pfclm.run4.out.clm_output.00002.C.pfb'

t3 = time.perf_counter()
pfb_data_in = np.zeros((3342, 1888, 17), order='F')
pf_fort_io.pfb_read(pfb_data_in, input_pfb)
pfb_data_in = np.transpose(pfb_data_in, (2, 1, 0))
t4 = time.perf_counter()
print(f'F load time {t4-t3:0.4f} seconds')

t1 = time.perf_counter()
pfb_read_rtn = pfio.pfread(input_pfb)
t2 = time.perf_counter()
print(f'C load time {t2-t1:0.4f} seconds')

np.testing.assert_array_equal(pfb_data_in, pfb_read_rtn)
