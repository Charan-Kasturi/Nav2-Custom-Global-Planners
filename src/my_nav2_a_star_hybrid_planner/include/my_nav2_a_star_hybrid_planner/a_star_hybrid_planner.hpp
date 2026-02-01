//#ifndef A_STAR_HYBRID_PLANNER_HPP
//#define A_STAR_HYBRID_PLANNER_HPP
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include <functional>           
#include "tf2_ros/buffer.h"
#include "std_msgs/msg/header.hpp"




#include "rclcpp/rclcpp.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"





namespace my_nav2_a_star_hybrid_planner
{

class a_star_hybrid_planner: public nav2_core::GlobalPlanner
{
    public:
        a_star_hybrid_planner() = default;
        ~a_star_hybrid_planner() override = default;

        void configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent,
        std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

        void cleanup() override;
        void activate() override;
        void deactivate() override;

        nav_msgs::msg::Path createPlan(const geometry_msgs::msg::PoseStamped &start,
                                       const geometry_msgs::msg::PoseStamped &goal) override;
                            
        private:
            struct pose
            {
                double x;
                double y;
                double theta;
            };
            struct cell
            {
                signed int x;
                signed int y;
                signed int theta;
            };
            struct cell_hash
            {
                std::size_t operator() (const cell &c) const noexcept
                {
                    return (static_cast<size_t> (c.x) << 32)^( static_cast<size_t> (c.y)) ^ static_cast<size_t>(c.theta);
                }
            };
            struct cell_eq
            {
                bool operator() (const cell &a , const cell &b) const noexcept
                {
                    return (a.x == b.x && a.y == b.y && a.theta == b.theta);
                }
            };
            bool world_to_map(double wx, double wy, signed int &mx, signed int &my) const;
            bool map_to_world(signed int mx, signed int my, double &wx, double &wy);
            double heuristic(const cell &a, const cell &b) const;
            bool is_lethal(unsigned char cost);
            nav_msgs::msg::Path reconstruct_path(const std::unordered_map<cell,cell,cell_hash,cell_eq> &parent,
                                                const cell &start,
                                                const cell &goal,
                                                const std_msgs::msg::Header &header);
            
            std::unordered_map< cell, std::vector<struct pose> ,cell_hash, cell_eq> edge_trace;
            ///params
            unsigned char lethal_cost_{253};
            int step_size_{30};
            int num_steps_{30};
            double each_step_cost_{1};

            //ros
            rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
            std::string name_;
            std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
            nav2_costmap_2d::Costmap2D * costmap_{nullptr};         

};
} // namespace my_nav2_a_star_hybrid_planner
//#endif