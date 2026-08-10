# Centralizes CMake policy settings for compatibility with CMake versions
# newer than cmake_minimum_required. Append new policies here to keep the main
# CMakeLists clean.
#
# Usage: include(NexusPolicies), then call nexus_apply_forward_policies().
#
# Note: a macro is used here rather than setting policies at the module top
# level. `cmake_policy(SET ...)` only affects the current scope — both
# functions and included files create new policy scopes whose settings are
# undone on return; a macro creates no scope and is inlined directly into the
# caller, so the settings persist and are inherited by subdirectories.

macro(nexus_apply_forward_policies)
  # CMP0219 (CMake >= 3.30): backslashes in macro call arguments are preserved
  # literally. HandleLLVMOptions passes source strings containing backslashes
  # to CHECK_C_SOURCE_COMPILES when probing compiler flags; set to NEW to
  # silence the warning.
  if(POLICY CMP0219)
    cmake_policy(SET CMP0219 NEW)
  endif()
endmacro()
