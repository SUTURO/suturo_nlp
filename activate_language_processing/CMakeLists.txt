cmake_minimum_required(VERSION 3.0.2)
project(activate_language_processing)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  suturo_msgs
  std_msgs
  rospy
)

# catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  suturo_msgs)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

