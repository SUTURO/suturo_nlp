cmake_minimum_required(VERSION 2.8.3)
project(suturo_textparser)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  tmc_rosjulius_msgs
)


catkin_package(

)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
