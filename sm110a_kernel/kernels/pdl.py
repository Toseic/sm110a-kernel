"""Shared PDL option validation and stable FC1 trigger point IDs."""

PDL_TRIGGER_POINTS = {
    "entry": 0,
    "post_init": 1,
    "mainloop_quarter": 2,
    "mainloop_mid": 3,
    "mainloop_three_quarters": 4,
    "mainloop_end": 5,
    "tail": 6,
}


def validate_pdl(pdl: bool) -> None:
    """Validate the shared compile-time input-wait/output-trigger switch."""
    if type(pdl) is not bool:
        raise TypeError("pdl must be bool")
