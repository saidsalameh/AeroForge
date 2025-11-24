# CMake generated Testfile for 
# Source directory: /Users/said/Library/CloudStorage/OneDrive-Personnel/Project Documents/AeroForge
# Build directory: /Users/said/Library/CloudStorage/OneDrive-Personnel/Project Documents/AeroForge/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(python_drone_nav_env "/opt/homebrew/Frameworks/Python.framework/Versions/3.10/bin/python3.10" "-m" "pytest" "-q" "tests/python/test_drone_nav_env.py")
set_tests_properties(python_drone_nav_env PROPERTIES  WORKING_DIRECTORY "/Users/said/Library/CloudStorage/OneDrive-Personnel/Project Documents/AeroForge" _BACKTRACE_TRIPLES "/Users/said/Library/CloudStorage/OneDrive-Personnel/Project Documents/AeroForge/CMakeLists.txt;33;add_test;/Users/said/Library/CloudStorage/OneDrive-Personnel/Project Documents/AeroForge/CMakeLists.txt;0;")
subdirs("external/bullet")
subdirs("external/pybind11")
subdirs("src/aeroforge_sim")
subdirs("src/bindings/python")
