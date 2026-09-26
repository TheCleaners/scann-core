# googletest `gtest_prod.h` (vendored)

ScaNN's headers include `gtest/gtest_prod.h` for its `FRIEND_TEST` macro
(which grants unit tests access to private members). That one header is
all scann-core needs from GoogleTest, so it is vendored rather than
fetching the whole framework.

* Source: https://github.com/google/googletest, release `v1.18.0`,
  `googletest/include/gtest/gtest_prod.h`, unmodified.
* License: BSD 3-Clause, see [LICENSE](LICENSE).

Without it, builds silently depended on GoogleTest being installed on the
build machine; a clean machine failed with "'gtest/gtest_prod.h' file not
found".
