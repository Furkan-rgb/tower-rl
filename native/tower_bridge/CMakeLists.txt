cmake_minimum_required(VERSION 3.22)
project(tower_bridge LANGUAGES CXX)

set(TOWER_BRIDGE_PACKAGE_VERSION "unconfigured" CACHE STRING "Official package version")
set(TOWER_BRIDGE_PACKAGE_VERSION_CODE "0" CACHE STRING "Official package version code")
set(TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256 "unconfigured" CACHE STRING "Official signer SHA-256")
set(TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256 "unconfigured" CACHE STRING "Original libunity SHA-256")
set(TOWER_BRIDGE_LIBIL2CPP_SHA256 "unconfigured" CACHE STRING "libil2cpp SHA-256")
set(TOWER_BRIDGE_UNITY_VERSION "6000.3.15f1" CACHE STRING "Unity version")
set(TOWER_BRIDGE_METADATA_VERSION "39" CACHE STRING "IL2CPP metadata version")
set(TOWER_BRIDGE_VERSION "tower-bridge-v1" CACHE STRING "Bridge version")
set(TOWER_BRIDGE_PROFILE_ID "unconfigured" CACHE STRING "Private training profile id")

option(TOWER_BRIDGE_DIAGNOSTICS "Log the semantic member inventory for a new game build" OFF)

add_library(tower_bridge SHARED tower_bridge.cpp)
target_compile_features(tower_bridge PRIVATE cxx_std_17)
target_compile_options(tower_bridge PRIVATE -Wall -Wextra -Werror -fvisibility=hidden)
target_link_libraries(tower_bridge PRIVATE dl log)

# The deployed artifact is identified by its own SHA-256 everywhere: the install
# directory is named for it, the deploy path re-reads it off the device, and the
# handshake is checked against it. That only means anything if the digest names
# the source rather than the directory it was built in, so nothing about this
# host's paths may reach the binary.
#
#   -ffile-prefix-map  rewrites the source directory to `.` in debug info and in
#                      every __FILE__-shaped string, so a build from a worktree
#                      and a build from the main checkout agree. (It is the union
#                      of -fdebug-prefix-map and -fmacro-prefix-map; the separate
#                      flags would be redundant.)
#   -ffile-compilation-dir  does the same for DW_AT_comp_dir, which is the build
#                      directory and is not covered by the prefix map above.
#   --strip-all        drops .symtab and the .debug_* sections outright. They are
#                      the only remaining carrier of absolute paths — the NDK
#                      sysroot's include paths land there — and nothing consumes
#                      them: the build tree is a mktemp directory that is gone by
#                      the time the bridge runs, and -fvisibility=hidden means
#                      the dynamic symbols a crash stack can name are kept.
#   --build-id=sha1    pins the note to a hash of the output's own contents
#                      rather than to whatever the linker defaults to, so the
#                      build id is a second witness of the same identity instead
#                      of an independent source of drift.
#
# The remaining inputs were checked and are not path-dependent: the source uses
# no __DATE__, __TIME__ or __FILE__; no define carries CMAKE_SOURCE_DIR; there is
# one translation unit and a fixed library order, so no archive or link ordering
# is at stake; there is no rpath, no linker map, and no LTO.
target_compile_options(tower_bridge PRIVATE
    -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.
    -ffile-compilation-dir=.
)
target_link_options(tower_bridge PRIVATE
    -Wl,--build-id=sha1
    -Wl,--strip-all
)

target_compile_definitions(tower_bridge PRIVATE
    TOWER_BRIDGE_PACKAGE_VERSION="${TOWER_BRIDGE_PACKAGE_VERSION}"
    TOWER_BRIDGE_PACKAGE_VERSION_CODE=${TOWER_BRIDGE_PACKAGE_VERSION_CODE}
    TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256="${TOWER_BRIDGE_OFFICIAL_SIGNER_SHA256}"
    TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256="${TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256}"
    TOWER_BRIDGE_LIBIL2CPP_SHA256="${TOWER_BRIDGE_LIBIL2CPP_SHA256}"
    TOWER_BRIDGE_UNITY_VERSION="${TOWER_BRIDGE_UNITY_VERSION}"
    TOWER_BRIDGE_METADATA_VERSION=${TOWER_BRIDGE_METADATA_VERSION}
    TOWER_BRIDGE_VERSION="${TOWER_BRIDGE_VERSION}"
    TOWER_BRIDGE_PROFILE_ID="${TOWER_BRIDGE_PROFILE_ID}"
)

if(TOWER_BRIDGE_DIAGNOSTICS)
  target_compile_definitions(tower_bridge PRIVATE TOWER_BRIDGE_DIAGNOSTICS=1)
endif()

# Render-off experiment (#27): render one player-loop frame in N. At 1 nothing is
# defined and the production artifact is byte-for-byte unchanged; a build above 1
# is an experiment variant selected only through TOWER_BRIDGE_BUILD_DIR, never
# installed as state/bridge/current and never used for spectate or recording.
set(TOWER_BRIDGE_RENDER_FRAME_INTERVAL "1" CACHE STRING
    "Render one player-loop frame in N (1 = every frame, production)")
# No leading zero: the value is pasted into C++ as a literal, where 016 is octal.
if(NOT TOWER_BRIDGE_RENDER_FRAME_INTERVAL MATCHES "^[1-9][0-9]*$" OR
   TOWER_BRIDGE_RENDER_FRAME_INTERVAL GREATER 60)
  message(FATAL_ERROR
          "TOWER_BRIDGE_RENDER_FRAME_INTERVAL must be an integer in 1..60, got "
          "'${TOWER_BRIDGE_RENDER_FRAME_INTERVAL}'")
endif()
if(TOWER_BRIDGE_RENDER_FRAME_INTERVAL GREATER 1)
  target_compile_definitions(tower_bridge PRIVATE
      TOWER_BRIDGE_RENDER_FRAME_INTERVAL=${TOWER_BRIDGE_RENDER_FRAME_INTERVAL})
endif()
