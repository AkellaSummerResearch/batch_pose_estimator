cmake_minimum_required(VERSION 2.8.3)
project(batch_pose_estimator)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  nav_msgs
  roscpp
  cmake_modules
  std_srvs
)


find_package(Eigen REQUIRED)

catkin_package(
  DEPENDS Eigen
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

add_executable(batch_solver src/batch_solver.cpp
                            src/msg_conversions.cpp)
target_link_libraries(batch_solver ${catkin_LIBRARIES})