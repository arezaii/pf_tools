# PF_FORT_IO

Python wrapped Fortran PF IO codes

## Install

Recommend that this is installed in it's own virtual environment, like a conda environment, rather than at the system level.

```
git clone https://github.com/arezaii/pf_tools.git
cd pf_tools/fort_io
pip install .
```

## Usage

Brief overview of the pf_read usage

```
>>>import pf_pytools.pf_fort_io as pf_fort_io
>>>import numpy as np
>>>pfb_data = np.asfortranarray(np.zeros((3342,1888,17)))
>>>pf_fort_io.pfb_read(pfb_data,'filename')
```

