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
