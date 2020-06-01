import click

from . import make_mask_cls
from . import make_mask_seg


@click.group()
def entry_point():
    pass


entry_point.add_command(make_mask_cls.make_mask_cls)
entry_point.add_command(make_mask_seg.make_mask_seg)
