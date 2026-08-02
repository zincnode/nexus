# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "NEXUS"

# We prefer the lit internal shell which provides a better user experience on
# failures unless the user explicitly disables it with LIT_USE_INTERNAL_SHELL=0
# env var.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

# Set the test format based on shell configuration.
config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.nexus_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# Tweak the PATH to include the tools dir. Only nexus tools are built in this
# tree; the FileCheck/not/count tools are resolved at runtime from the LLVM
# tools dir (config.llvm_tools_dir) via PATH.
config.nexus_tools_dir = os.path.join(config.nexus_obj_root, "bin")
tool_dirs = [config.nexus_tools_dir, config.llvm_tools_dir]
for dirs in tool_dirs:
    llvm_config.with_environment("PATH", dirs, append_path=True)

tools = [
    "nxs-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"

if config.enable_assertions:
    config.available_features.add("asserts")
else:
    config.available_features.add("noasserts")
