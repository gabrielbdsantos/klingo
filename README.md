# Klingo

A minimal framework for creating parameterized airfoil geometries.

## About

The core concept is simple:

  1. Parameterization based on array manipulation via [Numpy][numpy].
  2. Geometry generation via [numpy-stl][stl].

Klingo adopts the concept of _sections_ to define the blade geometry. Each
section consists of a two-dimensional airfoil profile. The various sections are
then grouped together to define the three-dimensional blade geometry. The
coordinates defining each section are stored using numpy, which makes ease to
perform a number of geometrical manipulations. Klingo already provides the most
common ones: _scale_, _translation_, and _rotation_.

Klingo also provides a simple method for exporting the final geometry to
`.stl`.

## Requirements

  * python >= 3.8
  * numpy >= 1.20
  * scipy >= 1.7
  * numpy-stl >= 2.16.3

## Install

Using [poetry][poetry]:

    poetry env use PATH_TO_YOUR_PYTHON_BINARY
    poetry install --no-dev

## TODO

  * [ ] Add tests
  * [ ] Add examples
  * [ ] Extend export capabilities

## License

This project is licensed under the MIT License. For further information refer
to [LICENSE][license].

[poetry]: https://python-poetry.org
[numpy]: https://www.numpy.org
[stl]: https://github.com/WoLpH/numpy-stl
[license]: ./LICENSE
