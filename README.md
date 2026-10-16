# libheric - A library for heterogeneous ERI computation

### Build Instructions
libheric depends on 4 Libraries:
 - [libint at commit f47820b](https://github.com/evaleev/libint/tree/f47820bd2b8eb1b71b3076249a8f516f1a5316e4)
 - [Eigen3 3.4.0](https://gitlab.com/libeigen/eigen/-/releases/3.4.0)
 - [vectorclass](https://github.com/vectorclass/version2)
 - [HiGHS 1.51.1](https://github.com/ERGO-Code/HiGHS/releases/tag/v1.15.1)

Eigen and vectorclass are header-only, but libint and HiGHS should be build as follows:
```bash
# libint
cmake -B build -DBUILD_SHARED_LIBS=ON -DLIBINT2_BUILD_SHARED_AND_STATIC_LIBS=ON -DLIBINT2_REQUIRE_CXX_API_COMPILED=OFF -DLIBINT2_API_PREFIX=libheric_ -DCMAKE_BUILD_TYPE=Release -DLIBINT2_ENABLE_ONEBODY=-1 -DLIBINT2_MULTIPOLE_MAX_ORDER=0 -DLIBINT2_ENABLE_G12DKH=-1 -DLIBINT2_ENABLE_ERI=1 -DLIBINT2_ENABLE_T1G12=OFF -DLIBINT2_ERI_MAX_AM="4;4" -DLIBINT2_ERI_OPT_AM="3;0" -DCMAKE_INSTALL_PREFIX=../libint_build -DLIBINT2_REALTYPE=Vec4d -DLIBINT2_USER_DEFINED_REAL_INCLUDES="#include <vectorclass.h>" -DCMAKE_CXX_FLAGS="-I/path/to/vectorclass/version2 -march=native" -DLIBINT2_ALIGN_SIZE=1 -DLIBINT2_ENABLE_UNROLLING=10000 -DLIBINT2_GENERATE_FMA=ON

# HiGHS
cmake . -B build  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_SHARED_EXTRAS_LIB=OFF -DBUILD_TESTING=OFF -DBUILD_CXX_EXE=OFF -DBUILD_EXAMPLES=OFF
```
Here, `LIBINT2_REALTYPE` is set to `Vec4d` for AVX2 vectorization, but other vectorclass datatypes should also work.
Also note that libint is namespaced, so that any library or program linking libheric can still link to a non-namespaced libint.
To reduce the compilation time of libint, set `LIBINT2_ERI_OPT_AM="2;0"`

libheric can then be build using
```bash
cmake -B build -Dvectorclass_ROOT=/path/to/version2 -Dlibint2_ROOT=/path/to/libint-build -DHiGHS_ROOT=/path/to/HiGHS-build -DCMAKE_BUILD_TYPE=Release

#Debug
cmake -B debug -Dvectorclass_ROOT=/path/to/version2 -Dlibint2_ROOT=/path/to/libint-build -DHiGHS_ROOT=/path/to/HiGHS-build -DCMAKE_BUILD_TYPE=Debug -DDEBUG_PRINT=ON
```

The `_ROOT` preprocessor flags are only necessary if the libraries are not installed in any folder on `PATH`. The `DEBUG_PRINT` flag in the debug build will enable log files, which will be stored per thread in `.libheric_logs` in the current working directory.
