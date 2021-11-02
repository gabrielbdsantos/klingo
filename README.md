# Klingo

A minimal framework for creating parameterized airfoil geometries.

## About

The core concept is simple:

  1. Parameterization based on array manipulation via [Numpy][numpy].
  2. Geometry generation via [OpenCascade][occ].

Klingo adopts the concept of _sections_ to define the blade geometry. Each
section consists of a two-dimensional airfoil profile. The various sections are
then grouped together to define the three-dimensional blade geometry. The
coordinates defining each section are stored using numpy, which makes ease to
perform a number of geometrical manipulations. Klingo already provides the most
common ones: _scale_, _translation_, and _rotation_.

Once the blade sections are defined, it is possible to create a solid object
using some of the klingo's OCC wrappers. For now, klingo only supports exporting
the final geometry to `.stl`.

## Requirements

  * python >= 3.8
  * numpy >= 1.20
  * scipy >= 1.7
  * pythonocc-core == 7.5.1

## Install :warning:

For now, poetry is not actually working. There is an issue going on with
hyphenated package names [python-poetry/poetry#4678][4678]. Thus, it is not
possible to add pythonocc-core as a dependencie.

## TODO

  * [ ] Add tests
  * [ ] Add examples
  * [ ] Extend export capabilities

## License

This project is licensed under the MIT License. For further information refer
to [LICENSE][license].

[numpy]: https://www.numpy.org
[occ]: https://www.opencascade.org
[license]: ./LICENSE
[4678]: https://github.com/python-poetry/poetry/issues/4678
