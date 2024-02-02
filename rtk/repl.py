"""
For quick debugging and testing.
"""

from rich import pretty, traceback

from rtk.utils import get_console


def install(
    max_depth: int = 3,
    max_length: int = 7,
    show_locals: bool = True,
    _console=get_console(),
    _pretty: bool = True,
    _traceback: bool = True,
):
    """
    Installs the rich traceback hook and pretty install.

    ## Args:
        `max_depth` (`int`, optional): Defaults to `3`.
        `max_length` (`int`, optional): Defaults to `7`.
        `show_locals` (`bool`, optional): Defaults to `True`.
    """
    if _pretty:
        pretty.install(max_depth=max_depth, max_length=max_length, console=_console)
    if _traceback:
        traceback.install(show_locals=show_locals, console=_console)


def prepare_console(**kwargs):
    from rtk.utils import login

    install(**kwargs)
    ws = login()
    console = get_console()
    console.clear()

    return ws, console
