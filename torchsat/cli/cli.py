import click

from . import make_mask_cls
from . import make_mask_seg
from . import calcuate_mean_std


@click.group()
def entry_point():
    pass


entry_point.add_command(make_mask_cls.make_mask_cls)
entry_point.add_command(make_mask_seg.make_mask_seg)
entry_point.add_command(calcuate_mean_std.calcuate_mean_std)
