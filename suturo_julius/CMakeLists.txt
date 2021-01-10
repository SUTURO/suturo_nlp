cmake_minimum_required(VERSION 2.8.3)
project(suturo_julius)

find_package(catkin REQUIRED COMPONENTS
  tmc_rosjulius
)

catkin_package(

)

include_directories(

)


# build julius dictionaries

execute_process(COMMAND mkdfa dictionaries/move_objs/move_objs
                OUTPUT_QUIET
                COMMAND mkdfa dictionaries/hard_commands/hard_commands
                OUTPUT_QUIET
                COMMAND mkdfa dictionaries/qa_system/qa_system
                OUTPUT_QUIET
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                
                )

