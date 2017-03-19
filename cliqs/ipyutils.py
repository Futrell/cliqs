""" Convenience functions for parallelization and visualization in IPython. """
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from .compat import *

import sys

def setup_pmap():
    try:
        import ipyparallel
        rc = ipyparallel.Client()
        rc[:].use_cloudpickle()
        cluster = rc.load_balanced_view()
        pmap = cluster.map
        return pmap
    except Exception as e:
        print(e, file=sys.stderr)
        return map
    
pmap = setup_pmap()
