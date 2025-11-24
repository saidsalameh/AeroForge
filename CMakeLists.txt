cmake_minimum_required(VERSION 3.16)
project(AeroForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Bullet configuration: disable demos / extras / tests ---
set(BUILD_BULLET2_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_BULLET3 OFF CACHE BOOL "" FORCE)
set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_OPENGL3_DEMOS OFF CACHE BOOL "" FORCE)

# --- Bullet as subdirectory ---
add_subdirectory(external/bullet)

# --- pybind11 ---
add_subdirectory(external/pybind11)

# Core simulation library
add_subdirectory(src/aeroforge_sim)

# Python bindings
add_subdirectory(src/bindings/python)

# Addind tests 

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Enable CTest
include(CTest)

add_test(
    NAME python_drone_nav_env
    COMMAND ${Python3_EXECUTABLE} -m pytest -q tests/python/test_drone_nav_env.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

