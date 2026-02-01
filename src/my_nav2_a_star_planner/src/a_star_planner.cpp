#include "my_nav2_a_star_planner/a_star_planner.hpp"

#include <cmath>
#include <queue>
#include <limits>

#include "nav2_util/node_utils.hpp"

namespace my_nav2_a_star_planner
{

void a_star_planner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer>,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();

  nav2_util::declare_parameter_if_not_declared(node_, name_ + ".allow_diagonal", rclcpp::ParameterValue(true));
  nav2_util::declare_parameter_if_not_declared(node_, name_ + ".prevent_corner_cutting", rclcpp::ParameterValue(true));
  nav2_util::declare_parameter_if_not_declared(node_, name_ + ".lethal_cost", rclcpp::ParameterValue(253));
  nav2_util::declare_parameter_if_not_declared(node_, name_ + ".cost_weight", rclcpp::ParameterValue(1.0));

  node_->get_parameter(name_ + ".allow_diagonal", allow_diagonal_);
  node_->get_parameter(name_ + ".prevent_corner_cutting", prevent_corner_cutting_);
  int lethal_int = 253;
  node_->get_parameter(name_ + ".lethal_cost", lethal_int);
  lethal_cost_ = static_cast<unsigned char>(lethal_int);
  node_->get_parameter(name_ + ".cost_weight", cost_weight_);
}

void a_star_planner::cleanup() {}
void a_star_planner::activate() {}
void a_star_planner::deactivate() {}

bool a_star_planner::is_lethal(unsigned char cost) const
{
  // treat unknown(255) as lethal too
  return (cost >= lethal_cost_);
}

bool a_star_planner::world_to_map(double wx, double wy, unsigned int & mx, unsigned int & my) const
{
  return costmap_->worldToMap(wx, wy, mx, my);
}

void a_star_planner::map_to_world(unsigned int mx, unsigned int my, double & wx, double & wy) const
{
  costmap_->mapToWorld(mx, my, wx, wy);
}

double a_star_planner::heuristic(const cell & a, const cell & b) const
{
  // octile distance (good for 8-connected grids)
  const double dx = std::abs(static_cast<double>(a.x) - static_cast<double>(b.x));
  const double dy = std::abs(static_cast<double>(a.y) - static_cast<double>(b.y));
  const double dmin = std::min(dx, dy);
  const double dmax = std::max(dx, dy);
  return (dmax - dmin) + std::sqrt(2.0) * dmin;
}

nav_msgs::msg::Path a_star_planner::reconstruct_path(
  const std::unordered_map<cell, cell, cell_hash, cell_eq> & parent,
  const cell & start_c,
  const cell & goal_c,
  const std_msgs::msg::Header & header)
{
  nav_msgs::msg::Path path;
  path.header = header;

  std::vector<cell> cells;
  cell cur = goal_c;
  cells.push_back(cur);

  while (!(cur.x == start_c.x && cur.y == start_c.y)) {
    auto it = parent.find(cur);
    if (it == parent.end()) {
      path.poses.clear();
      return path;
    }
    cur = it->second;
    cells.push_back(cur);
  }

  std::reverse(cells.begin(), cells.end());

  path.poses.reserve(cells.size());
  for (const auto & c : cells) {
    double wx, wy;
    map_to_world(c.x, c.y, wx, wy);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = header;
    ps.pose.position.x = wx;
    ps.pose.position.y = wy;
    ps.pose.position.z = 0.0;
    ps.pose.orientation.w = 1.0;
    path.poses.push_back(ps);
  }
  return path;
}

nav_msgs::msg::Path a_star_planner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  RCLCPP_WARN(
  node_->get_logger(),
  "A_STAR_PLANNER: createPlan() called (start=(%.2f, %.2f), goal=(%.2f, %.2f))",
  start.pose.position.x,
  start.pose.position.y,
  goal.pose.position.x,
  goal.pose.position.y
);

  const std::string global_frame = costmap_ros_->getGlobalFrameID();
  std_msgs::msg::Header out_header;
  out_header.frame_id = global_frame;
  out_header.stamp = node_->now();
  nav_msgs::msg::Path empty;
  empty.header = out_header;

  if (!costmap_) {
    RCLCPP_ERROR(node_->get_logger(), "Costmap is null");
    return empty;
  }

  // Nav2 expects global planner output in global frame
  if (start.header.frame_id != global_frame || goal.header.frame_id != global_frame) {
    RCLCPP_ERROR(
      node_->get_logger(),
      "Start/goal must be in global frame '%s' (got start='%s', goal='%s')",
      global_frame.c_str(), start.header.frame_id.c_str(), goal.header.frame_id.c_str());
    return empty;
  }

  unsigned int sx, sy, gx, gy;
  if (!world_to_map(start.pose.position.x, start.pose.position.y, sx, sy)) {
    RCLCPP_WARN(node_->get_logger(), "Start outside map");
    return empty;
  }
  if (!world_to_map(goal.pose.position.x, goal.pose.position.y, gx, gy)) {
    RCLCPP_WARN(node_->get_logger(), "Goal outside map");
    return empty;
  }

  const cell start_c{sx, sy};
  const cell goal_c{gx, gy};

  if (is_lethal(costmap_->getCost(sx, sy))) {
    RCLCPP_WARN(node_->get_logger(), "Start in lethal cell");
    return empty;
  }
  if (is_lethal(costmap_->getCost(gx, gy))) {
    RCLCPP_WARN(node_->get_logger(), "Goal in lethal cell");
    return empty;
  }

  // neighbor offsets
  struct step { int dx; int dy; double base; };
  std::vector<step> steps;
  steps.push_back({ 1, 0, 1.0});
  steps.push_back({-1, 0, 1.0});
  steps.push_back({ 0, 1, 1.0});
  steps.push_back({ 0,-1, 1.0});
  if (allow_diagonal_) {
    steps.push_back({ 1, 1, std::sqrt(2.0)});
    steps.push_back({ 1,-1, std::sqrt(2.0)});
    steps.push_back({-1, 1, std::sqrt(2.0)});
    steps.push_back({-1,-1, std::sqrt(2.0)});
  }

  // A* open set
  struct pq_item { double f; cell c; };
  struct pq_cmp { bool operator()(const pq_item & a, const pq_item & b) const { return a.f > b.f; } };
  std::priority_queue<pq_item, std::vector<pq_item>, pq_cmp> open;

  std::unordered_map<cell, double, cell_hash, cell_eq> g;
  std::unordered_map<cell, cell, cell_hash, cell_eq> parent;

  g[start_c] = 0.0;
  open.push({heuristic(start_c, goal_c), start_c});

  const int w = static_cast<int>(costmap_->getSizeInCellsX());
  const int h = static_cast<int>(costmap_->getSizeInCellsY());

  while (!open.empty()) {
    const cell cur = open.top().c;
    open.pop();

    if (cur.x == goal_c.x && cur.y == goal_c.y) {
      return reconstruct_path(parent, start_c, goal_c, out_header);
    }

    const double cur_g = g[cur];

    for (const auto & st : steps) {
      const int nx = static_cast<int>(cur.x) + st.dx;
      const int ny = static_cast<int>(cur.y) + st.dy;

      if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
        continue;
      }

      // corner cutting prevention for diagonals:
      // if moving (dx,dy) diagonally, both adjacent cardinals must be free
      if (prevent_corner_cutting_ && st.dx != 0 && st.dy != 0) {
        const unsigned char c1 = costmap_->getCost(static_cast<unsigned int>(cur.x + st.dx), cur.y);
        const unsigned char c2 = costmap_->getCost(cur.x, static_cast<unsigned int>(cur.y + st.dy));
        if (is_lethal(c1) || is_lethal(c2)) {
          continue;
        }
      }

      const unsigned char cell_cost = costmap_->getCost(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny));
      if (is_lethal(cell_cost)) {
        continue;
      }

      // Costmap costs are 0..255; scale them into a small additive penalty
      const double cost_penalty = cost_weight_ * (static_cast<double>(cell_cost) / 255.0);

      const cell nb{static_cast<unsigned int>(nx), static_cast<unsigned int>(ny)};
      const double tentative_g = cur_g + st.base + cost_penalty;

      auto it = g.find(nb);
      if (it == g.end() || tentative_g < it->second) {
        g[nb] = tentative_g;
        parent[nb] = cur;
        const double f = tentative_g + heuristic(nb, goal_c);
        open.push({f, nb});
      }
    }
  }

  RCLCPP_WARN(node_->get_logger(), "No path found");
  return empty;
}

}  // namespace my_nav2_a_star_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(my_nav2_a_star_planner::a_star_planner, nav2_core::GlobalPlanner)
