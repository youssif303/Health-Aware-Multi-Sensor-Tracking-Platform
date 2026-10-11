cmake_minimum_required(VERSION 3.16)
project(health_aware_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tracker src/tracker.cpp)
if(MSVC)
  target_compile_options(tracker PRIVATE /W4 /permissive-)
else()
  target_compile_options(tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()
