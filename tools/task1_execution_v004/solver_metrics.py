"""Expose SciPy TRF's existing iteration counter without altering its steps."""
import inspect,importlib,textwrap
def install():
    module=importlib.import_module("scipy.optimize._lsq.trf")
    fn=module.trf_bounds
    if getattr(fn,"reports_iterations",False):return
    source=textwrap.dedent(inspect.getsource(fn))
    assert source.count("return OptimizeResult(")==1
    source=source.replace("return OptimizeResult(", "return OptimizeResult(nit=iteration, ",1)
    scope=dict(module.__dict__)
    exec(compile(source,"<trf iteration telemetry>","exec"),scope)
    replacement=scope["trf_bounds"];replacement.reports_iterations=True
    module.trf_bounds=replacement
