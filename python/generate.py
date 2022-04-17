"""Generates instance inputs of small, medium, and large sizes.

Modify this file to generate your own problem instances.

For usage, run `python3 generate.py --help`.
"""

import argparse
from pathlib import Path
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
    print(cities)
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
    x=0
    for i in range(0, 100, 8):
        for j in range(0, 100, 7):
            if i%2:
                cities.append(Point(i, j+4))
            else:
                cities.append(Point(i,j))
            x = x+1
    
    # YOUR CODE HERE
    cities[7] = Point(36, 76)
    cities[9] = Point(89, 1)
    cities[14] = Point(12, 50)
    cities[25] = Point(64,62)
    cities[29] = Point(94,63)
    cities[34] = Point(53,83)
    cities[190] = Point(77, 49)
    cities[176] = Point(99, 26)
    cities[172] = Point(17,44)
    cities[166] = Point(52, 33)
    cities[165] = Point(73, 5)
    cities[161] = Point(31, 41)
    cities[160] = Point(16, 23)
    cities[153] = Point(22, 40)
    cities[151] = Point(92, 32)
    cities[144] = Point(3, 18)
    cities[143] = Point(87, 72)
    cities[139] = Point(48, 90)
    cities[135] = Point(88, 43)
    cities[134] = Point(54, 9)
    cities[107] = Point(96, 58)
    cities[36] = Point(86, 84)
    cities[40] = Point(74, 67)
    cities[43] = Point(35, 97)
    cities[44] = Point(29, 37)
    cities[73] = Point(34, 100)
    cities[79] = Point(28, 27)
    cities[83] = Point(47, 95)
    cities[87] = Point(91, 55)
    cities[98] = Point(81, 65)
    cities[103] = Point(6, 15)
    cities[105] = Point(30, 51)
    
    
    
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
