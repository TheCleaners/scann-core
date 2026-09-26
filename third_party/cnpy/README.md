# cnpy (vendored)

Reads and writes NumPy `.npy`/`.npz` files from C++. ScaNN uses it to load
and save index artifacts.

* Source: https://github.com/sammymax/cnpy (a fork of
  https://github.com/rogersce/cnpy by Carl Rogers), commit
  `57184ee0db37cac383fc29175950747a46a8b512` (2022-04-28), the commit
  upstream ScaNN's Bazel build pins.
* Files: `cnpy/cnpy.h` and `cnpy/cnpy.cpp`, unmodified. The rest of that
  repository (its own build files, tools and tests) is not needed.
* License: MIT, see [LICENSE](LICENSE).

Vendored rather than fetched so that building scann-core doesn't depend on
a personal fork staying online. The `cnpy` target is defined in
`cmake/Dependencies.cmake`.
