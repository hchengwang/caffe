cmake_minimum_required(VERSION 2.6.0)

project(april_tags)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME april_tags)
include(cmake/pods.cmake)
include(cmake/lcmtypes.cmake)
lcmtypes_build()

