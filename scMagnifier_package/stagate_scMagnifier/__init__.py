# stagate_scMagnifier/__init__.py
__version__ = "0.1.0"

__all__ = ["spatial_preprocess","spatial_perturb","spatial_consensus","spatial_merge"]  


def __getattr__(name):

    import_map = {
        "spatial_preprocess": ("spatial_preprocess_core", "spatial_preprocess"),
        "spatial_perturb": ("spatial_perturb_core", "spatial_perturb"),
        "spatial_consensus": ("spatial_consensus_core", "spatial_consensus"),
        "spatial_merge": ("spatial_merge_core", "spatial_merge"),
    }
    

    if name in import_map:
        module_name, attr_name = import_map[name]

        module = __import__(f"stagate_scMagnifier.{module_name}", fromlist=[attr_name])

        return getattr(module, attr_name)
    

    raise AttributeError(f"module 'stagate_scMagnifier' has no attribute '{name}'")