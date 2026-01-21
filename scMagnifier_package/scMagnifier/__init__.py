# scMagnifier/__init__.py
__version__ = "0.1.0"
__all__ = [
    "preprocess", "GRN", "perturb", "consensus", "merge", "TFscore",
    "multi_preprocess","multi_perturb","multi_consensus","multi_merge",
    "spatial_preperturb","umap_compare"
]  


def __getattr__(name):
    
    import_map = {
        "preprocess": ("preprocess_core", "preprocess"),
        "GRN": ("grn_core", "GRN"),
        "perturb": ("perturb_core", "perturb"),
        "consensus": ("consensus_core", "consensus"),
        "merge": ("merge_core", "merge"),
        "TFscore": ("TFscore_core", "TFscore"),
        "multi_preprocess": ("multi_preprocess_core", "multi_preprocess"),
        "multi_perturb": ("multi_perturb_core", "multi_perturb"),
        "multi_consensus": ("multi_consensus_core", "multi_consensus"),
        "multi_merge": ("multi_merge_core", "multi_merge"),
        "spatial_preperturb": ("spatial_preperturb_core", "spatial_preperturb"),
        "umap_compare": ("umap_compare_core", "umap_compare"),
    }
    
    if name in import_map:
        
        module_name, attr_name = import_map[name]
        module = __import__(f"scMagnifier.{module_name}", fromlist=[attr_name])
        return getattr(module, attr_name)
    
    
    raise AttributeError(f"module 'scMagnifier' has no attribute '{name}'")