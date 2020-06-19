import unittest
import numpy as np
import pf_pytools.pf_fort_io as fort_io
import pfio


class ReaderTestVerification(unittest.TestCase):
    def test_something(self):
        numpy_data = np.load('lw.press.init.npy')
        input_pfb = 'press.init.pfb'
        fort_pfb_data = np.zeros((41, 41, 50), order='F')
        fort_io.pfb_read(fort_pfb_data, input_pfb)
        fort_pfb_data = np.transpose(fort_pfb_data, (2, 1, 0))
        c_pfb_data = pfio.pfread(input_pfb)
        self.assertIsNone(np.testing.assert_array_equal(fort_pfb_data, c_pfb_data))
        self.assertIsNone(np.testing.assert_array_equal(fort_pfb_data, numpy_data))
        self.assertIsNone(np.testing.assert_array_equal(c_pfb_data, numpy_data))


if __name__ == '__main__':
    unittest.main()
