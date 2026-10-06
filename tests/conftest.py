"""Pytest global configuration and environment setup."""

# Patch transformers is_datasets_available and check_torch_load_is_safe
try:
    import transformers.utils as _u
    import transformers.utils.import_utils as _iu
    import transformers.modeling_utils as _mu
    _iu.is_datasets_available = lambda: False
    _u.is_datasets_available = lambda: False
    _iu.check_torch_load_is_safe = lambda: None
    _mu.check_torch_load_is_safe = lambda: None
except Exception:
    pass
