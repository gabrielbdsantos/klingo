# Klingo

A minimal framework for creating parameterized airfoil geometries.

## About

The core concept is simple:

  1. Parameterization based on array manipulation via [numpy][numpy].
  2. Geometry generation via [numpy-stl][stl].

Klingo adopts the concept of _sections_ to define the blade geometry. Each
section consists of a two-dimensional airfoil with respect to some plane in the
three-dimensional space. The various sections are then grouped together to
define the three-dimensional blade geometry. The coordinates defining each
section are stored using numpy, which makes ease to perform a number of
geometrical manipulations. Klingo already provides the most common ones:
_scale_, _translation_, and _rotation_.

Klingo also provides a simple method for exporting the final geometry to
`.stl`.

## Install

### Poetry

    $ poetry add git+https://github.com/gabrielbdsantos/klingo.git@master

### Pip

    $ pip install git+https://github.com/gabrielbdsantos/klingo.git@master

## TODO

  * [ ] Add examples
  * [ ] Add tests

## License

This project is licensed under the MIT License. For further information see the
[LICENSE][license].

[numpy]: https://www.numpy.org
[stl]: https://github.com/WoLpH/numpy-stl
[license]: ./LICENSE
