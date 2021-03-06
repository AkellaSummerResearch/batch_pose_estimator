
#include "ros/ros.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_srvs/Trigger.h>

#include <mg_msgs/RequestRelativePoseBatch.h>

#include "batch_pose_estimator/msg_conversions.h"

// typedef message_filters::sync_policies::ExactTime<nav_msgs::Odometry, geometry_msgs::PoseStamped> SyncPolicy;
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, geometry_msgs::PoseStamped> SyncPolicyApprox;
typedef message_filters::Synchronizer<SyncPolicyApprox> Sync;

class BatchPoseSolver {
  ros::NodeHandle nh_;
  std::string in_pose_topic_, in_slam_topic_;
  std::string new_batch_srv_name_;
  std::vector<boost::shared_ptr<Sync> > sync_;
  std::vector<message_filters::Subscriber<nav_msgs::Odometry>*> pose_sub_;
  std::vector<message_filters::Subscriber<geometry_msgs::PoseStamped>*> slam_sub_;
  std::vector<std::pair<nav_msgs::Odometry, geometry_msgs::PoseStamped> > pose_pair_vec_;
  std::vector<std::string> namespaces_;
  // std::vector<Eigen::Quaterniond> pose_att_vec;
  // std::vector<Eigen::Vector3d> pose_pos_vec;
  int n_elements_;
  bool get_data_;
  ros::Publisher pose_pub_, slam_pub;
  ros::ServiceServer start_new_batch_srv_;
  geometry_msgs::Pose rel_pose_;
  Eigen::Vector3d cam_pos_body_frame_;

 public:
  BatchPoseSolver(ros::NodeHandle *nh) {
    nh_= *nh;
    nh_.getParam("input_pose_topic", in_pose_topic_);
    nh_.getParam("input_slam_topic", in_slam_topic_);
    nh_.getParam("namespaces", namespaces_);

    // Subscribe to position measurements and slam measurements
    sync_.resize(namespaces_.size());
    for (uint i = 0; i < namespaces_.size(); i++) {
      std::string pose_topic = "/" + namespaces_[i] + in_pose_topic_;
      std::string slam_topic = "/" + namespaces_[i] + in_slam_topic_;
      pose_sub_.push_back(new message_filters::Subscriber<nav_msgs::Odometry>(nh_, pose_topic, 1));
      slam_sub_.push_back(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh_, slam_topic, 1));
      sync_[i].reset(new Sync(SyncPolicyApprox(10), *pose_sub_[i], *slam_sub_[i]));
      sync_[i]->registerCallback(boost::bind(&BatchPoseSolver::PoseCallback, this, _1, _2));
      ROS_INFO("[batch_pose_estimator] Input pose topic: %s", pose_sub_[i]->getTopic().c_str());
      ROS_INFO("[batch_pose_estimator] Input slam topic: %s", slam_sub_[i]->getTopic().c_str());
    }

    // Set default number of elements
    n_elements_ = 100;

    // Get cam position in body frame
    std::vector<double> cam_pos;
    nh_.getParam("cam_pos_body_frame", cam_pos);
    cam_pos_body_frame_ = Eigen::Vector3d(cam_pos[0], cam_pos[1], cam_pos[2]);

    // Service for starting a new batch problem
    new_batch_srv_name_ = "start_new_batch";
    start_new_batch_srv_ = nh_.advertiseService(new_batch_srv_name_,
                                                &BatchPoseSolver::StartNewBatch, this);

    // Debug publishers
    pose_pub_ = nh_.advertise<nav_msgs::Odometry>("pose", 1000);
    slam_pub = nh_.advertise<geometry_msgs::PoseStamped>("slam", 1000);

    // Set callback to capture data for a batch estimation
    get_data_ = false;

  }

  void PoseCallback(const nav_msgs::Odometry::ConstPtr& pose_msg,
                    const geometry_msgs::PoseStamped::ConstPtr& slam_msg) {
    // ROS_INFO("%s", pose_msg->child_frame_id.c_str());
    if (!get_data_) {
      return;
    }

    // Save the pair into a vector
    Eigen::Quaterniond q_enu = msg_conversions::ros_to_eigen_quat(pose_msg->pose.pose.orientation);
    Eigen::Vector3d pos_w = 
      q_enu.toRotationMatrix()*cam_pos_body_frame_ + msg_conversions::ros_point_to_eigen_vector(pose_msg->pose.pose.position);
    nav_msgs::Odometry odom = *pose_msg;
    odom.pose.pose.position = msg_conversions::eigen_to_ros_point(pos_w);
    std::pair<nav_msgs::Odometry, geometry_msgs::PoseStamped> pose_pair(odom, *slam_msg);
    pose_pair_vec_.push_back(pose_pair);

    // // Create vectors of pose orientation and translation
    // pose_att_vec.push_back(msg_conversions::ros_to_eigen_quat(pose_msg->pose.pose.orientation));
    // pose_pos_vec.push_back(msg_conversions::ros_point_to_eigen_vector(pose_msg->pose.pose.position));

    // Debug prints
    // ROS_INFO("dt = %f", (pose_msg->header.stamp - slam_msg->header.stamp).toSec());
    // ROS_INFO("count = %zd", pose_pair_vec_.size());
    printf("\rNumber of pose/slam pairs = %zd", pose_pair_vec_.size());
    std::fflush(stdout);

    // If there are more than the minimum number of elements, solve for relative pose
    if (pose_pair_vec_.size() >= n_elements_) {
      ROS_INFO("[batch_pose_estimator] Solving for relative pose...");

      // Relative orientation between the frames
      Eigen::Quaterniond q_rel = this->SolveRelativeOrientation(pose_pair_vec_);
      Eigen::Matrix3d rot = q_rel.toRotationMatrix().transpose();
      std::cout << "Relative rotation: " << std::endl << rot << std::endl;

      // Relative position between the frames
      Eigen::Vector3d pos = this->SolveRelativePosition(pose_pair_vec_, rot);
      std::cout << "Relative position: " << std::endl << pos << std::endl << std::endl;

      // // Mean pose of the drone throughout the batch solution
      // Eigen::Vector3d mean_pos = this->MeanPosition(pose_pos_vec);
      // Eigen::Quaterniond mean_att = this->AverageQuaternion(pose_att_vec);

      // Update results
      rel_pose_.position = msg_conversions::eigen_to_ros_point(pos);
      rel_pose_.orientation = msg_conversions::eigen_to_ros_quat(q_rel.inverse());

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
      rel_orientations.push_back(rot2*rot1.inverse());
      // Eigen::Quaterniond quat = rot2*rot1.inverse();
      // std::cout << quat.w() << " " << quat.x() << " " << quat.y() << " " << quat.z() << std::endl;
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

  Eigen::Vector3d MeanPosition(const std::vector<Eigen::Vector3d> &pos_vec) {
    Eigen::Vector3d sum_positions = Eigen::Vector3d::Zero();
    for (uint i = 0; i < pos_vec.size(); i++) {
      sum_positions = pos_vec[i];
    }

    return sum_positions/pos_vec.size();
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

  bool StartNewBatch(mg_msgs::RequestRelativePoseBatch::Request &req,
                     mg_msgs::RequestRelativePoseBatch::Response &res) {
    if (req.data > 0) {  // If requested number is <1, then n_elements_ takes its default value of 100
      n_elements_ = req.data;
    }
    pose_pair_vec_.clear();
    get_data_ = true;

    ROS_INFO("[batch_pose_estimator] Starting new relative pose batch with %d measurements!", n_elements_);

    // Wait until result is obtained
    ros::Rate loop_rate(10);
    while (get_data_ == true) {
      ros::spinOnce();
      loop_rate.sleep();
    }

    // Return the calculated relative pose
    res.pose = rel_pose_;
    return true;
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "relative_pose_solver");
  ros::NodeHandle node("~");

  BatchPoseSolver solver(&node);

  ros::spin();

  return 0;
}