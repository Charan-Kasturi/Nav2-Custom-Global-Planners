cmake_minimum_required(VERSION 3.8)
project(my_nav2_a_star_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(include)

add_library(my_nav2_a_star_planner SHARED
  src/a_star_planner.cpp
)

ament_target_dependencies(my_nav2_a_star_planner
  rclcpp
  nav2_core
  nav2_util
  nav2_costmap_2d
  nav_msgs
  geometry_msgs
  pluginlib
  tf2_ros
)

pluginlib_export_plugin_description_file(nav2_core plugin.xml)

install(TARGETS my_nav2_a_star_planner
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

install(FILES plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
