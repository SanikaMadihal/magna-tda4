# Install script for directory: /home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/aarch64-linux-gnu-objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/simd/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/src/md5/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/libturbojpeg.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xbinx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "tjbench" FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/tjbench-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xincludex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/turbojpeg.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/libjpeg.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xbinx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "cjpeg" FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/cjpeg-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xbinx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "djpeg" FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/djpeg-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xbinx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM RENAME "jpegtran" FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/jpegtran-static")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xbinx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/rdjpgcom")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/aarch64-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/rdjpgcom")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xbinx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/wrjpgcom")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/aarch64-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/wrjpgcom")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdocx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/libjpeg-turbo" TYPE FILE FILES
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/README.ijg"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/README.md"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/example.c"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/tjcomp.c"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/tjdecomp.c"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/tjtran.c"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/libjpeg.txt"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/structure.txt"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/usage.txt"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/wizard.txt"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/LICENSE.md"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmanx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1" TYPE FILE FILES
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/cjpeg.1"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/djpeg.1"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/jpegtran.1"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/rdjpgcom.1"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/doc/wrjpgcom.1"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/pkgscripts/libjpeg.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/pkgscripts/libturbojpeg.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/libjpeg-turbo" TYPE FILE FILES
    "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/pkgscripts/libjpeg-turboConfig.cmake"
    "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/pkgscripts/libjpeg-turboConfigVersion.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libjpeg-turbo/libjpeg-turboTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libjpeg-turbo/libjpeg-turboTargets.cmake"
         "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/CMakeFiles/Export/lib/cmake/libjpeg-turbo/libjpeg-turboTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libjpeg-turbo/libjpeg-turboTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libjpeg-turbo/libjpeg-turboTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/libjpeg-turbo" TYPE FILE FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/CMakeFiles/Export/lib/cmake/libjpeg-turbo/libjpeg-turboTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/libjpeg-turbo" TYPE FILE FILES "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/CMakeFiles/Export/lib/cmake/libjpeg-turbo/libjpeg-turboTargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xincludex" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/jconfig.h"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/jerror.h"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/jmorecfg.h"
    "/home/sanika/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/turbojpeg-sys-1.1.1/libjpeg-turbo/src/jpeglib.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/sanika/mi-isal-main/target/aarch64-unknown-linux-gnu/release/build/turbojpeg-sys-6655f6cadfd85278/out/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
