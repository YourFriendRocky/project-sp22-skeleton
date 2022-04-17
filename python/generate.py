"""Generates instance inputs of small, medium, and large sizes.

Modify this file to generate your own problem instances.

For usage, run `python3 generate.py --help`.
"""

import argparse
from pathlib import Path
from random import random
from typing import Callable, Dict

from instance import Instance
from size import Size
from point import Point
from file_wrappers import StdoutFileWrapper


def make_small_instance() -> Instance:
    """Creates a small problem instance.

    Size.SMALL.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
           

    cities = []
    x = 0
    y = 0
    while (x <= 30):        
        if (x % 6) == 1:
             y += 3
        while (y <= 30):
            cities.append(Point(x,y))
            y += 7
        y = 0
        x += 6
    cities[7] = Point(14, 16)
    cities[8] = Point(9,7)
    cities[12] = Point(26,3)
    cities[16] = Point(21,1)
    cities[18] = Point(17,0)
    cities[24] = Point(22,5)
    return Size.SMALL.instance(cities)


def make_medium_instance() -> Instance:
    """Creates a medium problem instance.

    Size.MEDIUM.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
    cities = []
    x = [29,0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 48, 8, 8, 8, 8, 16, 38, 37, 16, 13, 3, 16, 16, 24, 24, 24, 24, 18, 24, 24, 24, 32, 16, 25, 32, 32, 32, 32, 32, 40, 40, 15, 40, 40, 40, 40, 40, 48,48, 48, 48, 48, 48, 48]
    y = [7, 7, 14, 21, 28, 35, 42, 49, 0, 7, 14, 28, 28, 35, 42, 49, 0, 17, 22, 21, 49, 26, 42, 49, 0, 7, 14, 21,30, 35, 42, 49, 0, 1, 43, 21, 28, 35, 42, 49, 0, 7, 35, 21, 28, 35, 42, 49, 0, 7, 14, 21, 28, 35, 42]
    for i in range(55):
        cities.append(Point(x[i], y[i]))
    return Size.MEDIUM.instance(cities)


def make_large_instance() -> Instance:
    """Creates a large problem instance.

    Size.LARGE.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
    cities = []
    x = 0
    y = 0
    while (x <= 100):
        if ()
    # YOUR CODE HERE
    cities.append(Point(0, 0))
    cities.append(Point(0, 7))
    cities.append(Point(0, 14))
    cities.append(Point(0, 21))
    cities.append(Point(0, 28))
    cities.append(Point(0, 35))
    cities.append(Point(0, 42))
    cities.append(Point(0, 56))
    cities.append(Point(0, 70))
    cities.append(Point(0, 77))
    cities.append(Point(0, 84))
    cities.append(Point(8, 0))
    cities.append(Point(8, 7))
    cities.append(Point(8, 14))
    cities.append(Point(8, 21))
    cities.append(Point(8, 28))
    cities.append(Point(8, 35))
    cities.append(Point(8, 42))
    cities.append(Point(8, 42))
    cities.append(Point(8, 49))
    cities.append(Point(8, 56))
    cities.append(Point(8, 63))
    cities.append(Point(8, 77))
    cities.append(Point(8, 84))
    cities.append(Point)(8, 91)
    cities.append(Point(16, 0))
    cities.append(Point(16, 7))
    cities.append(Point(16, 14))
    cities.append(Point(16, 21))
    cities.append(Point(16, 35))
    cities.append(Point(16, 49))
    cities.append(Point(16, 56))
    cities.append(Point(16, 63))
    cities.append(Point(16, 77))
    cities.append(Point(16, 84))
    cities.append(Point(24, 0))
    cities.append(Point(24, 7))
    cities.append(Point(24, 14))
    cities.append(Point(24, 21))
    cities.append(Point(24, 28))
    cities.append(Point(24, 35))
    cities.append(Point(24, 42))
    cities.append(Point(24, 49))
    cities.append(Point(24, 56))
    cities.append(Point(24, 63))
    cities.append(Point(24, 70))
    cities.append(Point(24, 77))
    cities.append(Point(24, 84))
    cities.append(Point(24, 91))
    cities.append(Point(24, 98))
    cities.append(Point(32, 0))
    cities.append(Point(32, 7))
    cities.append(Point(32, 14))
    cities.append(Point(32, 21))
    cities.append(Point(32, 28))
    cities.append(Point(32, 35))
    cities.append(Point(32, 42))
    cities.append(Point(32, 49))
    cities.append(Point(32, 56))
    cities.append(Point(32, 63))
    cities.append(Point(32, 70))
    cities.append(Point(32, 77))
    cities.append(Point(32, 84))
    cities.append(Point(32, 98))
    cities.append(Point(40, 0))
    cities.append(Point(40, 7))
    cities.append(Point(40, 14))
    cities.append(Point(40, 21))
    cities.append(Point(40, 35))
    return Size.LARGE.instance(cities)


# You shouldn't need to modify anything below this line.
SMALL = 'small'
MEDIUM = 'medium'
LARGE = 'large'

SIZE_STR_TO_GENERATE: Dict[str, Callable[[], Instance]] = {
    SMALL: make_small_instance,
    MEDIUM: make_medium_instance,
    LARGE: make_large_instance,
}

SIZE_STR_TO_SIZE: Dict[str, Size] = {
    SMALL: Size.SMALL,
    MEDIUM: Size.MEDIUM,
    LARGE: Size.LARGE,
}

def outfile(args, size: str):
    if args.output_dir == "-":
        return StdoutFileWrapper()

    return (Path(args.output_dir) / f"{size}.in").open("w")


def main(args):
    for size, generate in SIZE_STR_TO_GENERATE.items():
        if size not in args.size:
            continue

        with outfile(args, size) as f:
            instance = generate()
            assert instance.valid(), f"{size.upper()} instance was not valid."
            assert SIZE_STR_TO_SIZE[size].instance_has_size(instance), \
                f"{size.upper()} instance did not meet size requirements."
            print(f"# {size.upper()} instance.", file=f)
            instance.serialize(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate problem instances.")
    parser.add_argument("output_dir", type=str, help="The output directory to "
                        "write generated files to. Use - for stdout.")
    parser.add_argument("--size", action='append', type=str,
                        help="The input sizes to generate. Defaults to "
                        "[small, medium, large].",
                        default=None,
                        choices=[SMALL, MEDIUM, LARGE])
    # action='append' with a default value appends new flags to the default,
    # instead of creating a new list. https://bugs.python.org/issue16399
    args = parser.parse_args()
    if args.size is None:
        args.size = [SMALL, MEDIUM, LARGE]
    main(args)
