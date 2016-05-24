""" Convenience functions for parallelization and visualization in IPython. """

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
