cmake_minimum_required(VERSION 3.0.2)

project(suturo_nlg)

find_package(catkin REQUIRED COMPONENTS message_generation)


generate_messages(
    DEPENDENCIES
)



catkin_package(
  CATKIN_DEPENDS message_runtime)


include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)
