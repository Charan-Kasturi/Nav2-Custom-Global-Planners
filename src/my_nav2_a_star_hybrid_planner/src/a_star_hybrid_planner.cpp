#include "my_nav2_a_star_hybrid_planner/a_star_hybrid_planner.hpp"

#include <cmath>
#include <queue>
#include <limits>
#include <algorithm>
#include <unordered_set>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace my_nav2_a_star_hybrid_planner
{

void a_star_hybrid_planner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer>,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_ ? costmap_ros_->getCostmap() : nullptr;

  node_->declare_parameter(name_ + ".lethal_cost", lethal_cost_);
  node_->declare_parameter(name_ + ".step_size", step_size_);
  node_->declare_parameter(name_ + ".num_steps", num_steps_);
  node_->declare_parameter(name_ + ".each_step_cost", each_step_cost_);

  // New tunables (kept local, no header changes needed)
  node_->declare_parameter(name_ + ".min_turn_radius_cells", 30.0);
  node_->declare_parameter(name_ + ".goal_vicinity_cells", 2);
  node_->declare_parameter(name_ + ".theta_bins", 72);

  node_->get_parameter(name_ + ".lethal_cost", lethal_cost_);
  node_->get_parameter(name_ + ".step_size", step_size_);
  node_->get_parameter(name_ + ".num_steps", num_steps_);
  node_->get_parameter(name_ + ".each_step_cost", each_step_cost_);
}

void a_star_hybrid_planner::cleanup()
{
  edge_trace.clear();
}

void a_star_hybrid_planner::activate() {}
void a_star_hybrid_planner::deactivate() {}

bool a_star_hybrid_planner::world_to_map(double wx, double wy, signed int & mx, signed int & my) const
{
  if (!costmap_) {
    return false;
  }

  unsigned int umx = 0;
  unsigned int umy = 0;

  if (!costmap_->worldToMap(wx, wy, umx, umy)) {
    return false;
  }

  mx = static_cast<signed int>(umx);
  my = static_cast<signed int>(umy);
  return true;
}

bool a_star_hybrid_planner::map_to_world(signed int mx, signed int my, double & wx, double & wy)
{
  if (!costmap_) {
    return false;
  }
  if (mx < 0 || my < 0) {
    return false;
  }

  costmap_->mapToWorld(static_cast<unsigned int>(mx), static_cast<unsigned int>(my), wx, wy);
  return true;
}

double a_star_hybrid_planner::heuristic(const cell & a, const cell & b) const
{
  const double dx = static_cast<double>(a.x - b.x);
  const double dy = static_cast<double>(a.y - b.y);
  return std::hypot(dx, dy);
}

bool a_star_hybrid_planner::is_lethal(unsigned char cost)
{
  return cost >= lethal_cost_;
}

nav_msgs::msg::Path a_star_hybrid_planner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  RCLCPP_INFO(node_->get_logger(), "MY HYBRID A* PLANNER IS RUNNING");

  nav_msgs::msg::Path out;
  out.header = start.header;
 
  if (!node_ || !costmap_) {
    RCLCPP_ERROR(rclcpp::get_logger("a_star_hybrid_planner"), "Planner not configured or costmap is null");
    return out;
  }

  // Read tunables (declared in configure)
  double min_turn_radius_cells = 30.0;
  int goal_vicinity_cells = 10;
  int theta_bins = 72;

  node_->get_parameter(name_ + ".min_turn_radius_cells", min_turn_radius_cells);
  node_->get_parameter(name_ + ".goal_vicinity_cells", goal_vicinity_cells);
  node_->get_parameter(name_ + ".theta_bins", theta_bins);

  if (theta_bins <= 0) {
    theta_bins = 72;
  }
  if (goal_vicinity_cells <= 0) {
    goal_vicinity_cells = 10;
  }
  if (min_turn_radius_cells < 1.0) {
    min_turn_radius_cells = 1.0;
  }

  const double two_pi = 2.0 * M_PI;
  const double theta_bin_size = two_pi / static_cast<double>(theta_bins);

  const int sx_cells = static_cast<int>(costmap_->getSizeInCellsX());
  const int sy_cells = static_cast<int>(costmap_->getSizeInCellsY());

  const double res = costmap_->getResolution();
  const double origin_x = costmap_->getOriginX();
  const double origin_y = costmap_->getOriginY();

  auto normalize_angle = [&](double a) -> double {
    a = std::fmod(a, two_pi);
    if (a < 0.0) {
      a += two_pi;
    }
    return a;
  };

  auto theta_to_bin = [&](double theta) -> int {
    const double t = normalize_angle(theta);
    int b = static_cast<int>(t / theta_bin_size);
    if (b < 0) b = 0;
    if (b >= theta_bins) b = theta_bins - 1;
    return b;
  };

  auto cell_to_world_continuous = [&](double cx, double cy, double & wx, double & wy) {
    wx = origin_x + (cx + 0.5) * res;
    wy = origin_y + (cy + 0.5) * res;
  };

  signed int s_mx = 0, s_my = 0, g_mx = 0, g_my = 0;
  if (!world_to_map(start.pose.position.x, start.pose.position.y, s_mx, s_my) ||
      !world_to_map(goal.pose.position.x, goal.pose.position.y, g_mx, g_my))
  {
    RCLCPP_WARN(node_->get_logger(), "Start or goal is outside the costmap");
    return out;
  }

  if (s_mx < 0 || s_my < 0 || g_mx < 0 || g_my < 0 ||
      s_mx >= sx_cells || s_my >= sy_cells || g_mx >= sx_cells || g_my >= sy_cells)
  {
    RCLCPP_WARN(node_->get_logger(), "Start or goal map indices invalid");
    return out;
  }

  if (is_lethal(costmap_->getCost(static_cast<unsigned int>(s_mx), static_cast<unsigned int>(s_my))) ||
      is_lethal(costmap_->getCost(static_cast<unsigned int>(g_mx), static_cast<unsigned int>(g_my))))
  {
    RCLCPP_WARN(node_->get_logger(), "Start or goal in lethal cell");
    return out;
  }

  const double start_yaw = tf2::getYaw(start.pose.orientation);
  const double goal_yaw  = tf2::getYaw(goal.pose.orientation);

  pose start_pose;
  start_pose.x = static_cast<double>(s_mx);
  start_pose.y = static_cast<double>(s_my);
  start_pose.theta = normalize_angle(start_yaw);

  pose goal_pose;
  goal_pose.x = static_cast<double>(g_mx);
  goal_pose.y = static_cast<double>(g_my);
  goal_pose.theta = normalize_angle(goal_yaw);

  const cell start_key{
    static_cast<signed int>(std::lround(start_pose.x)),
    static_cast<signed int>(std::lround(start_pose.y)),
    static_cast<signed int>(theta_to_bin(start_pose.theta))
  };

  const cell goal_key{
    static_cast<signed int>(std::lround(goal_pose.x)),
    static_cast<signed int>(std::lround(goal_pose.y)),
    static_cast<signed int>(theta_to_bin(goal_pose.theta))
  };

  std::unordered_map<cell, double, cell_hash, cell_eq> g_cost;
  std::unordered_map<cell, cell, cell_hash, cell_eq> parent;

  // Closed set to avoid re-expanding same discrete state
  std::unordered_set<cell, cell_hash, cell_eq> closed;

  edge_trace.clear();

  struct open_item
  {
    double f;
    cell key;
    pose continuous;
  };

  struct open_cmp
  {
    bool operator()(const open_item & a, const open_item & b) const
    {
      return a.f > b.f;
    }
  };

  std::priority_queue<open_item, std::vector<open_item>, open_cmp> open;

  g_cost[start_key] = 0.0;
  open.push(open_item{heuristic(start_key, goal_key), start_key, start_pose});

  auto propagate = [&](const pose & p0, double curvature_mult,
                       pose & p1, std::vector<pose> & trace_out) -> bool
  {
    trace_out.clear();

    if (num_steps_ <= 0) {
      return false;
    }

    const double ds = static_cast<double>(step_size_) / static_cast<double>(num_steps_);  // cell units
    double x = p0.x;
    double y = p0.y;
    double th = p0.theta;

    for (int i = 0; i < num_steps_; ++i) {
      x += ds * std::cos(th);
      y += ds * std::sin(th);

      // curvature_mult = 0 means straight
      if (curvature_mult != 0.0) {
        th += (ds / min_turn_radius_cells) * curvature_mult;
      }
      th = normalize_angle(th);

      const int ix = static_cast<int>(std::lround(x));
      const int iy = static_cast<int>(std::lround(y));

      if (ix < 0 || iy < 0 || ix >= sx_cells || iy >= sy_cells) {
        return false;
      }

      const unsigned char cost = costmap_->getCost(
        static_cast<unsigned int>(ix),
        static_cast<unsigned int>(iy));

      if (is_lethal(cost)) {
        return false;
      }

      pose ps;
      cell_to_world_continuous(x, y, ps.x, ps.y);
      ps.theta = th;
      trace_out.push_back(ps);
    }

    p1.x = x;
    p1.y = y;
    p1.theta = th;
    return true;
  };

  cell final_key = start_key;
  bool found = false;

  while (!open.empty()) {
    open_item cur = open.top();
    open.pop();
    //if (cur_g > g_cost[cur_key] + eps) continue;


    const cell cur_key = cur.key;

    // Skip if already expanded
    if (closed.find(cur_key) != closed.end()) {
      continue;
    }
    closed.insert(cur_key);

    const pose cur_pose = cur.continuous;

    // Goal check: position only
    const double dxg = goal_pose.x - cur_pose.x;
    const double dyg = goal_pose.y - cur_pose.y;
    const double dist_cells = std::hypot(dxg, dyg);

    if (dist_cells < static_cast<double>(goal_vicinity_cells)) {
      final_key = cur_key;
      found = true;
      break;
    }

    // 5 steering options instead of 3:
    // hard left, left, straight, right, hard right
    const double curvatures[5] = {+2.0, +1.0, 0.0, -1.0, -2.0};

    for (int m = 0; m < 5; ++m) {
      const double curv = curvatures[m];

      pose next_pose;
      std::vector<pose> trace_out;

      if (!propagate(cur_pose, curv, next_pose, trace_out)) {
        continue;
      }

      const int nx = static_cast<int>(std::lround(next_pose.x));
      const int ny = static_cast<int>(std::lround(next_pose.y));
      if (nx < 0 || ny < 0 || nx >= sx_cells || ny >= sy_cells) {
        continue;
      }

      const cell next_key{nx, ny, theta_to_bin(next_pose.theta)};

      if (closed.find(next_key) != closed.end()) {
        continue;
      }

      const unsigned char cell_cost = costmap_->getCost(
        static_cast<unsigned int>(nx),
        static_cast<unsigned int>(ny));

      const double cost_term = static_cast<double>(cell_cost) / 255.0;

      // Add a small curvature penalty so it does not over-turn
      const double curvature_penalty = std::fabs(curv) * 0.05;

      const double new_g = g_cost[cur_key] +
                           static_cast<double>(step_size_) * each_step_cost_ +
                           cost_term +
                           curvature_penalty;

      auto it = g_cost.find(next_key);
      if (it == g_cost.end() || new_g < it->second) {
        g_cost[next_key] = new_g;
        parent[next_key] = cur_key;
        edge_trace[next_key] = trace_out;

        const double f = new_g + heuristic(next_key, goal_key);
        open.push(open_item{f, next_key, next_pose});
      }
    }
  }

  if (!found) {
    RCLCPP_WARN(node_->get_logger(), "Hybrid A* failed to find a path");
    return out;
  }

  return reconstruct_path(parent, start_key, final_key, start.header);
}

nav_msgs::msg::Path a_star_hybrid_planner::reconstruct_path(
  const std::unordered_map<cell,cell,cell_hash,cell_eq> & parent,
  const cell & start,
  const cell & goal,
  const std_msgs::msg::Header & header)
{
  nav_msgs::msg::Path path;
  path.header = header;

  std::vector<geometry_msgs::msg::PoseStamped> reversed_poses;

  cell cur = goal;

  const std::size_t max_backtrack = 500000;
  std::size_t steps = 0;

  while (!(cur.x == start.x && cur.y == start.y && cur.theta == start.theta)) {
    if (++steps > max_backtrack) {
      RCLCPP_ERROR(node_->get_logger(), "Backtrack overflow, parent chain likely broken");
      break;
    }

    auto tr_it = edge_trace.find(cur);
    if (tr_it != edge_trace.end()) {
      const std::vector<pose> & tr = tr_it->second;

      for (auto rit = tr.rbegin(); rit != tr.rend(); ++rit) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = header;
        ps.pose.position.x = rit->x;
        ps.pose.position.y = rit->y;
        ps.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, rit->theta);
        ps.pose.orientation = tf2::toMsg(q);

        reversed_poses.push_back(ps);
      }
    }

    auto p_it = parent.find(cur);
    if (p_it == parent.end()) {
      RCLCPP_ERROR(node_->get_logger(), "Parent missing during reconstruction");
      break;
    }
    cur = p_it->second;
  }

  std::reverse(reversed_poses.begin(), reversed_poses.end());
  path.poses = std::move(reversed_poses);
  return path;
}

}  // namespace my_nav2_a_star_hybrid_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(my_nav2_a_star_hybrid_planner::a_star_hybrid_planner, nav2_core::GlobalPlanner)
