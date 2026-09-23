cmake_minimum_required(VERSION 3.20)
project(gpt2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O2 -Wall -Wextra)

set(LLMBRICKS_BUILD_TESTS OFF CACHE BOOL "Build llm-bricks tests" FORCE)
set(LLMBRICKS_INSTALL OFF CACHE BOOL "Install llm-bricks" FORCE)
add_subdirectory(thirdparty/llm-bricks EXCLUDE_FROM_ALL)

add_executable(gpt2
    main.cpp
    csrc/ops.cpp
    csrc/tokenizer.cpp
    csrc/weights.cpp
    csrc/modeling.cpp
)

target_link_libraries(gpt2 PRIVATE llm_bricks::llm_bricks)
