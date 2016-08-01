find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(rv-cv-bridge-lcm SHARED rv-cv-bridge-lcm.cpp)
target_link_libraries(rv-cv-bridge-lcm ${OpenCV_LIBS} 
    bot2-core z
    lcmtypes_bot2-core lcmtypes_kinect)
pods_install_libraries(rv-cv-bridge-lcm)
pods_install_headers(cv_bridge_lcm/rv-cv-bridge-lcm.h DESTINATION cv_bridge_lcm)
pods_install_pkg_config_file(rv-cv-bridge-lcm 
    LIBS -lrv-cv-bridge-lcm
    REQUIRES opencv
    VERSION 0.0.1)

add_executable(rv-cv-images-lcm rv-cv-images-lcm.cpp)
pods_use_pkg_config_packages(rv-cv-images-lcm lcm bot2-core)
target_link_libraries(rv-cv-images-lcm 
    rv-cv-bridge-lcm  
    ${Boost_FILESYSTEM_LIBRARY} boost_system boost_filesystem
    )
pods_install_executables(rv-cv-images-lcm)

