""" Convenience functions for parallelization and visualization in IPython. """
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

def setup_pmap():
    try:
        import ipyparallel
        rc = ipyparallel.Client()
        rc[:].use_cloudpickle()
        cluster = rc.load_balanced_view()
        pmap = cluster.map
        return pmap
    except Exception as e:
        print(e)
        return map
    
pmap = setup_pmap()
