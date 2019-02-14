
#include "ros/ros.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_srvs/Trigger.h>

#include "batch_pose_estimator/msg_conversions.h"

// typedef message_filters::sync_policies::ExactTime<nav_msgs::Odometry, geometry_msgs::PoseStamped> SyncPolicy;
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, geometry_msgs::PoseStamped> SyncPolicyApprox;
typedef message_filters::Synchronizer<SyncPolicyApprox> Sync;

class BatchPoseSolver {
  ros::NodeHandle nh_;
  std::string in_pose_topic_, in_slam_topic_, out_batch_result_topic_;
  std::string new_batch_srv_name_;
  boost::shared_ptr<Sync> sync_;
  message_filters::Subscriber<nav_msgs::Odometry> pose_sub_;
  message_filters::Subscriber<geometry_msgs::PoseStamped> slam_sub_;
  std::vector<std::pair<nav_msgs::Odometry, geometry_msgs::PoseStamped> > pose_pair_vec_;
  int min_elements_;
  bool get_data_;
  ros::Publisher rel_pose_pub_;
  ros::Publisher pose_pub_, slam_pub;
  ros::ServiceServer start_new_batch_srv_;

 public:
  BatchPoseSolver(ros::NodeHandle *nh) {
    nh_= *nh;
    nh_.getParam("input_pose_topic", in_pose_topic_);
    nh_.getParam("input_slam_topic", in_slam_topic_);
    nh_.getParam("out_topic", out_batch_result_topic_);
    nh_.getParam("min_elements", min_elements_);

    // Subscribe to position measurements and slam measurements
    pose_sub_.subscribe(nh_, in_pose_topic_, 1);
    slam_sub_.subscribe(nh_, in_slam_topic_, 1);
    sync_.reset(new Sync(SyncPolicyApprox(10), pose_sub_, slam_sub_));
    sync_->registerCallback(boost::bind(&BatchPoseSolver::PoseCallback, this, _1, _2));

    // Service for starting a new batch problem
    new_batch_srv_name_ = "start_new_batch";
    start_new_batch_srv_ = nh_.advertiseService(new_batch_srv_name_,
                                                &BatchPoseSolver::StartNewBatch, this);

    // Publisher for relative pose
    rel_pose_pub_ = nh_.advertise<nav_msgs::Odometry>(out_batch_result_topic_, 1000);
    pose_pub_ = nh_.advertise<nav_msgs::Odometry>("pose", 1000);
    slam_pub = nh_.advertise<geometry_msgs::PoseStamped>("slam", 1000);

    // Set callback to capture data for a batch estimation
    get_data_ = true;

  }

  void PoseCallback(const nav_msgs::Odometry::ConstPtr& pose_msg,
                    const geometry_msgs::PoseStamped::ConstPtr& slam_msg) {
    if (!get_data_) {
      return;
    }

    // Save the pair into a vector
    std::pair<nav_msgs::Odometry, geometry_msgs::PoseStamped> pose_pair(*pose_msg, *slam_msg);
    pose_pair_vec_.push_back(pose_pair);

    // Debug prints
    // ROS_INFO("dt = %f", (pose_msg->header.stamp - slam_msg->header.stamp).toSec());
    // ROS_INFO("count = %zd", pose_pair_vec_.size());

    // If there are more than the minimum number of elements, solve for relative pose
    if (pose_pair_vec_.size() >= min_elements_) {
      ROS_INFO("Solving for relative pose...");

      // Relative orientation between the frames
      Eigen::Quaterniond q_rel = this->SolveRelativeOrientation(pose_pair_vec_);
      Eigen::Matrix3d rot = q_rel.toRotationMatrix().transpose();
      std::cout << "Relative rotation: " << std::endl << rot << std::endl;

      // Relative position between the frames
      Eigen::Vector3d pos = this->SolveRelativePosition(pose_pair_vec_, rot);
      std::cout << "Relative position: " << std::endl << pos << std::endl << std::endl;

      // Publish results
      geometry_msgs::Pose rel_pose;
      rel_pose.position = msg_conversions::eigen_to_ros_point(pos);
      rel_pose.orientation = msg_conversions::eigen_to_ros_quat(q_rel);

      // Clear vector of measurements
      pose_pair_vec_.clear();
      get_data_ = false;
    }

    pose_pub_.publish(*pose_msg);
    slam_pub.publish(*slam_msg);
  }

  Eigen::Quaterniond SolveRelativeOrientation(const std::vector<std::pair<nav_msgs::Odometry, geometry_msgs::PoseStamped> > &pose_pair_vec) {
    std::vector<Eigen::Quaterniond> rel_orientations;
    for (uint i = 0; i < pose_pair_vec.size(); i++) {
      Eigen::Quaterniond rot1 = 
        msg_conversions::ros_to_eigen_quat(pose_pair_vec[i].first.pose.pose.orientation);
      Eigen::Quaterniond rot2 = 
        msg_conversions::ros_to_eigen_quat(pose_pair_vec[i].second.pose.orientation);
      rel_orientations.push_back(rot2.inverse()*rot1);
    }

    return this->AverageQuaternion(rel_orientations);
  }

  Eigen::Vector3d SolveRelativePosition(const std::vector<std::pair<nav_msgs::Odometry, geometry_msgs::PoseStamped> > &pose_pair_vec,
                                        const Eigen::Matrix3d &relative_rotation) {
    // The optimal solution is the mean of relative positions
    Eigen::Vector3d sum_positions = Eigen::Vector3d::Zero();
    for (uint i = 0; i < pose_pair_vec.size(); i++) {
      Eigen::Vector3d pos_meas = 
        msg_conversions::ros_point_to_eigen_vector(pose_pair_vec[i].first.pose.pose.position);
      Eigen::Vector3d pos_slam = 
        msg_conversions::ros_point_to_eigen_vector(pose_pair_vec[i].second.pose.position);
      sum_positions = sum_positions + (pos_meas - relative_rotation*pos_slam);
    }

    return sum_positions/pose_pair_vec.size();
  }

  // Algorithm can be seen in:
  // http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
  Eigen::Quaterniond AverageQuaternion(const std::vector<Eigen::Quaterniond> &quat_list) {
    Eigen::MatrixXd M(4,4);
    Eigen::MatrixXd quat(4,1);
    M = Eigen::MatrixXd::Zero(4,4);

    for (uint i = 0; i < quat_list.size(); i++) {
      quat << quat_list[i].w(), quat_list[i].x(), quat_list[i].y(), quat_list[i].z();
      M = M + quat*quat.transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU);

    Eigen::MatrixXd U = svd.matrixU();
    // std::cout << "Singular Vectors:" << std::endl << svd.matrixU() << std::endl;

    return Eigen::Quaterniond(U(0,0), U(1,0), U(2,0), U(3,0));
  }

  bool StartNewBatch(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) {
    get_data_ = true;
    pose_pair_vec_.clear();
    res.message = "Starting new batch!";
    res.success = true;
    return true;
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "batch_solver");
  ros::NodeHandle node("~");

  BatchPoseSolver solver(&node);

  ros::spin();

  return 0;
}