file(GLOB_RECURSE lcm_srcs "${PROJECT_SOURCE_DIR}/lcm/*.cpp")

foreach(source_file ${lcm_srcs})
  message(STATUS "${source_file}")

  # get file name
  get_filename_component(name ${source_file} NAME_WE)
    
  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)
  
  find_library(rv-cv-bridge-lcm_LIBRARIES rv-cv-bridge-lcm ~/caffe/build/lib)
  find_library(bot2-core_LIBRARIES bot2-core ~/caffe/build/lib)

  add_executable(${name} ${source_file})
  target_link_libraries(${name} ${Caffe_LINK} 
  	lcm
  	${rv-cv-bridge-lcm_LIBRARIES} 
  	${bot2-core_LIBRARIES})
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lcm/${folder}")

  caffe_set_solution_folder(${name} lcm)

  # install
  install(TARGETS ${name} DESTINATION bin)

  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standartaized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/lcm/${folder}/${name}${Caffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
endforeach()
