# 集中管理 CMake 策略设置，用于兼容比 cmake_minimum_required 更新的 CMake。
# 新策略在此追加即可，保持主 CMakeLists 干净。
#
# 用法：include(NexusPolicies) 之后调用 nexus_apply_forward_policies()。
#
# 注意：这里用宏（macro）而非在模块顶层直接设置策略。`cmake_policy(SET ...)`
# 只作用于当前 scope——函数与 include 的文件都会新建策略作用域，其中的设置会在
# 返回时被撤销；而宏不创建作用域，直接内联到调用方，因此设置能持久生效并被子
# 目录继承。

macro(nexus_apply_forward_policies)
  # CMP0219（CMake >= 3.30）：宏调用参数中的反斜杠按字面保留。
  # HandleLLVMOptions 探测编译器标志时会给 CHECK_C_SOURCE_COMPILES 传入含反斜杠
  # 的源码字符串；设为 NEW 以消除该 warning。
  if(POLICY CMP0219)
    cmake_policy(SET CMP0219 NEW)
  endif()
endmacro()
