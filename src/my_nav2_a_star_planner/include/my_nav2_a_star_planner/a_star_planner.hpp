#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace my_nav2_a_star_planner
{

class a_star_planner : public nav2_core::GlobalPlanner
{
public:
  a_star_planner() = default;
  ~a_star_planner() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  struct cell
  {
    unsigned int x;
    unsigned int y;
  };

  struct cell_hash
  {
    std::size_t operator()(const cell & c) const noexcept
    {
      // simple hash
      return (static_cast<std::size_t>(c.x) << 32) ^ static_cast<std::size_t>(c.y);
    }
  };

  struct cell_eq
  {
    bool operator()(const cell & a, const cell & b) const noexcept
    {
      return a.x == b.x && a.y == b.y;
    }
  };

  bool world_to_map(double wx, double wy, unsigned int & mx, unsigned int & my) const;
  void map_to_world(unsigned int mx, unsigned int my, double & wx, double & wy) const;

  double heuristic(const cell & a, const cell & b) const;

  bool is_lethal(unsigned char cost) const;

  nav_msgs::msg::Path reconstruct_path(
    const std::unordered_map<cell, cell, cell_hash, cell_eq> & parent,
    const cell & start_c,
    const cell & goal_c,
    const std_msgs::msg::Header & header);

  // params
  bool allow_diagonal_{true};
  bool prevent_corner_cutting_{true};
  unsigned char lethal_cost_{253};  // nav2 convention: 254/255 often lethal/unknown, keep configurable
  double cost_weight_{1.0};         // weight of costmap cost added to step cost

  // ros
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::string name_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
};

}  // namespace my_nav2_a_star_planner
