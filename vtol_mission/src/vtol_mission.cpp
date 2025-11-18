#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/srv/set_mode.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include "mavros_msgs/srv/command_tol.hpp" 

using namespace std::chrono_literals;

class BasicTakeoffLandNode : public rclcpp::Node
{
public:
    BasicTakeoffLandNode() : Node("vtol_mission_node")
    {
        mission_state_ = MissionState::WAIT_FOR_COMMAND;
        
        state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
            "/mavros/state", 10, std::bind(&BasicTakeoffLandNode::stateCallback, this, std::placeholders::_1));
        
        local_pos_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mavros/local_position/pose", 10, std::bind(&BasicTakeoffLandNode::localPositionCallback, this, std::placeholders::_1));

        start_sub_ = this->create_subscription<std_msgs::msg::Empty>(
            "/mission/start", 10, std::bind(&BasicTakeoffLandNode::startCallback, this, std::placeholders::_1));

        arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");
        takeoff_client_ = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");
        
        timer_ = this->create_wall_timer(100ms, std::bind(&BasicTakeoffLandNode::controlLoop, this));

        RCLCPP_INFO(get_logger(), "Basic Takeoff-Land Node started.");
        RCLCPP_INFO(get_logger(), "Waiting for FCU connection and local position...");
    }

private:
    enum class MissionState {
        WAIT_FOR_COMMAND, 
        SET_GUIDED_MODE,
        ARM,
        REQUEST_TAKEOFF, // (오타 수정됨)
        CLIMBING,
        HOLDING,
        SET_LAND_MODE,
        LANDING,
        DONE
    };
    MissionState mission_state_;

    mavros_msgs::msg::State current_state_;
    geometry_msgs::msg::PoseStamped current_pose_;
    bool local_pos_valid_ = false;
    bool fcu_connected_ = false;

    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr start_sub_;
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pos_sub_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr takeoff_client_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Time hold_start_time_;

    void stateCallback(const mavros_msgs::msg::State::SharedPtr msg) {
        current_state_ = *msg;
        fcu_connected_ = msg->connected;
    }

    void localPositionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_pose_ = *msg;
        local_pos_valid_ = true;
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, 
                             "Current Position (XYZ): [%.2f, %.2f, %.2f]",
                             current_pose_.pose.position.x,
                             current_pose_.pose.position.y,
                             current_pose_.pose.position.z);
    }

    void startCallback(const std_msgs::msg::Empty::SharedPtr msg)
    {
        (void)msg;
        if (mission_state_ == MissionState::WAIT_FOR_COMMAND)
        {
            if (fcu_connected_ && local_pos_valid_) {
                RCLCPP_INFO(get_logger(), "[ACTION] 'start' command received. Initiating mission.");
                mission_state_ = MissionState::SET_GUIDED_MODE;
            } else {
                RCLCPP_WARN(get_logger(), "'start' command received, but system not ready (No GPS or FCU connection).");
            }
        } else {
            RCLCPP_WARN(get_logger(), "Mission already in progress. 'start' command ignored.");
        }
    }

    void setMode(const std::string &mode) {
        auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        request->custom_mode = mode;
        if (!set_mode_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(get_logger(), "SetMode service not available"); return;
        }
        set_mode_client_->async_send_request(request);
    }

    void arm() {
        auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        request->value = true;
        if (!arming_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(get_logger(), "Arming service not available"); return;
        }
        arming_client_->async_send_request(request);
    }

    void requestTakeoff(float altitude_m) {
        auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
        request->altitude = altitude_m;
        if (!takeoff_client_->wait_for_service(1s)) {
            RCLCPP_ERROR(get_logger(), "Takeoff service not available"); return;
        }
        takeoff_client_->async_send_request(request);
    }

    void controlLoop()
    {
        switch (mission_state_) {
        case MissionState::WAIT_FOR_COMMAND:
            if (!fcu_connected_ || !local_pos_valid_) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "System not ready. Waiting for FCU connection and GPS (local position)...");
            } else {
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000, "System READY. Waiting for 'start' command on /mission/start topic.");
            }
            break;
        case MissionState::SET_GUIDED_MODE:
            if (current_state_.mode != "GUIDED") {
                RCLCPP_INFO(get_logger(), "[ACTION] Setting GUIDED mode...");
                setMode("GUIDED");
            } else {
                mission_state_ = MissionState::ARM;
            }
            break;
        case MissionState::ARM:
            if (!current_state_.armed) {
                RCLCPP_INFO(get_logger(), "[ACTION] Sending ARM command...");
                arm();
            } else {
                mission_state_ = MissionState::REQUEST_TAKEOFF;
            }
            break;
        case MissionState::REQUEST_TAKEOFF:
            RCLCPP_INFO(get_logger(), "[ACTION] Requesting Takeoff to 50m...");
            requestTakeoff(50.0f);
            mission_state_ = MissionState::CLIMBING;
            break;
        case MissionState::CLIMBING:
            if (current_pose_.pose.position.z >= 49.0) {
                RCLCPP_INFO(get_logger(), "[STATUS] Altitude reached (%.2f m). Holding for 5 seconds.", current_pose_.pose.position.z);
                hold_start_time_ = this->now();
                mission_state_ = MissionState::HOLDING;
            }
            break;
        case MissionState::HOLDING:
            if ((this->now() - hold_start_time_) >= 5s) {
                RCLCPP_INFO(get_logger(), "[STATUS] Hold complete (5s). [ACTION] Setting LAND mode...");
                mission_state_ = MissionState::SET_LAND_MODE;
            }
            break;
        case MissionState::SET_LAND_MODE:
            if (current_state_.mode != "LAND") {
                setMode("LAND");
            } else {
                mission_state_ = MissionState::LANDING;
            }
            break;
        case MissionState::LANDING:
            if (!current_state_.armed && current_pose_.pose.position.z < 1.0) {
                RCLCPP_INFO(get_logger(), "[SUCCESS] Mission complete. Landed and Disarmed.");
                mission_state_ = MissionState::DONE;
                rclcpp::shutdown();
            }
            break;
        case MissionState::DONE:
            break;
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BasicTakeoffLandNode>());
    rclcpp::shutdown();
    return 0;
}