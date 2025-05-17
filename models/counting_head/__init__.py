
from .HRSegment import build_counting_head as HRSegment


def build_counting_head(args):
    if args.name == "HRSegMent":
        return HRSegment(args)
        
    raise NotImplementedError("{} is not supported".format(args.name))