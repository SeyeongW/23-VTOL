// file: vtol_auto_waypoint_node.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"

#include "px4_msgs/msg/vehicle_status.hpp"
#include "px4_msgs/msg/vehicle_local_position.hpp"
#include "px4_msgs/msg/vehicle_command.hpp"
#include "px4_msgs/msg/home_position.hpp"
#include "px4_msgs/msg/offboard_control_mode.hpp"
#include "px4_msgs/msg/trajectory_setpoint.hpp"

using namespace std::chrono_literals;

static inline double deg2rad(double d){ return d * M_PI / 180.0; }
static inline double rad2deg(double r){ return r * 180.0 / M_PI; }

class VtolAutoWaypointNode : public rclcpp::Node
{
public:
    VtolAutoWaypointNode() : Node("vtol_auto_waypoint_node")
    {
        using std::placeholders::_1;

        // ---------------- Parameters ----------------
        takeoff_alt_m_       = this->declare_parameter<double>("takeoff_alt_m",        20.0);
        v_req_ms_            = this->declare_parameter<double>("forward_req_speed_ms", 12.0); // transition speed condition
        a_limit_g_           = this->declare_parameter<double>("transition_acc_g_max", 0.30); // max |a| during transition
        xy_tol_m_            = this->declare_parameter<double>("xy_tol_m",             6.0);
        z_tol_m_             = this->declare_parameter<double>("z_tol_m",              3.0);
        ctrl_period_ms_      = this->declare_parameter<int>("control_period_ms",       50);   // 20 Hz
        progress_log_ms_     = this->declare_parameter<int>("progress_log_ms",         1500);

        // ---------------- PX4 I/O ----------------
        status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
            "/fmu/out/vehicle_status", 10, std::bind(&VtolAutoWaypointNode::statusCb, this, _1));

        lpos_sub_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", 10, std::bind(&VtolAutoWaypointNode::lposCb, this, _1));

        home_sub_ = this->create_subscription<px4_msgs::msg::HomePosition>(
            "/fmu/out/home_position", 10, std::bind(&VtolAutoWaypointNode::homeCb, this, _1));

        cmd_pub_      = this->create_publisher<px4_msgs::msg::VehicleCommand>       ("/fmu/in/vehicle_command",       10);
        offboard_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode> ("/fmu/in/offboard_control_mode", 10);
        traj_pub_     = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>  ("/fmu/in/trajectory_setpoint",   10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(ctrl_period_ms_),
            std::bind(&VtolAutoWaypointNode::controlLoop, this));

        RCLCPP_INFO(get_logger(),
                    "VTOL auto mission sequence: WP1(MC) -> transition -> FW(WP2-5) -> back WP2 -> FW->MC -> WP1 -> HOME LAND");

        // ---------------- Waypoints (LOCAL NED) ----------------
        // NED frame: z is Down(+). To fly at altitude H, set z = -H.
        wp1_ = {0,   0.0,  60.0, -takeoff_alt_m_,  0.0f};
        wp2_ = {0,  30.0, 100.0, -takeoff_alt_m_, 45.0f};
        wp3_ = {0,   0.0, 140.0, -takeoff_alt_m_,  0.0f};
        wp4_ = {0, -30.0, 180.0, -takeoff_alt_m_, -45.0f};
        wp5_ = {0,   0.0, 220.0, -takeoff_alt_m_,  0.0f};

        // Initialize velocity history for acceleration estimation
        last_vx_ = NAN; last_vy_ = NAN; last_v_ts_ = this->now();
    }

private:
    // ---------- Type definitions ----------
    struct Wp {
        int frame;        // 0: LOCAL(NED)
        double x, y, z;   // meters (North, East, Down)
        float yaw_deg;
    };

    enum class St {
        WAIT = 0,
        ARM, TAKEOFF,
        MC_TO_WP1,
        CHECK_TRANSITION_COND,   // (v >= v_req) && (|a_horiz| <= limit)
        TRANSITION_MC2FW,
        FW_WP2, FW_WP3, FW_WP4, FW_WP5,
        BACK_TO_WP2,
        TRANSITION_FW2MC,
        MC_TO_WP1_BACK,
        LAND_HOME,
        DONE
    };

    // ---------- PX4 state ----------
    uint8_t arming_state_{0};
    bool lpos_valid_{false};
    double x_{0}, y_{0}, z_{0}, vx_{0}, vy_{0};
    bool home_valid_{false};
    double home_lat_{0}, home_lon_{0}, home_alt_{0};

    // acceleration estimate
    double last_vx_{NAN}, last_vy_{NAN};
    rclcpp::Time last_v_ts_;

    // ---------- mission ----------
    Wp wp1_, wp2_, wp3_, wp4_, wp5_;
    St st_{St::WAIT};
    rclcpp::Time last_log_ts_;

    // ---------- parameters ----------
    double takeoff_alt_m_{20.0};
    double v_req_ms_{12.0};
    double a_limit_g_{0.3};
    double xy_tol_m_{6.0};
    double z_tol_m_{3.0};
    int    ctrl_period_ms_{50};
    int    progress_log_ms_{1500};

    // ---------- ROS handles ----------
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr         status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr  lpos_sub_;
    rclcpp::Subscription<px4_msgs::msg::HomePosition>::SharedPtr          home_sub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr           cmd_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr      offboard_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr       traj_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ---------- utility functions ----------
    uint64_t nowUsec() const { return this->get_clock()->now().nanoseconds()/1000; }

    void publishOffboardKeepalive(bool pos=true) {
        px4_msgs::msg::OffboardControlMode m{};
        m.timestamp = nowUsec();
        m.position      = pos;
        m.velocity      = false;
        m.acceleration  = false;
        m.attitude      = false;
        m.body_rate     = false;
        offboard_pub_->publish(m);
    }

    void publishTrajSetpoint(double x, double y, double z, double yaw_deg) {
        px4_msgs::msg::TrajectorySetpoint sp{};
        sp.timestamp   = nowUsec();
        sp.position    = { (float)x, (float)y, (float)z };
        sp.yaw         = (float)deg2rad(yaw_deg);
        traj_pub_->publish(sp);
    }

    void sendCmd(uint16_t cmd, float p1=0,float p2=0,float p3=0,float p4=0,
                 float p5=0,float p6=0,float p7=0){
        px4_msgs::msg::VehicleCommand c{};
        c.timestamp = nowUsec();
        c.param1=p1; c.param2=p2; c.param3=p3; c.param4=p4;
        c.param5=p5; c.param6=p6; c.param7=p7;
        c.command = cmd;
        c.target_system=1; c.target_component=1;
        c.source_system=1; c.source_component=1;
        c.from_external = true;
        cmd_pub_->publish(c);
    }

    bool reachedLocal(const Wp &wp) const {
        double dx=x_-wp.x, dy=y_-wp.y, dz=z_-wp.z;
        return (std::hypot(dx,dy) < xy_tol_m_) && (std::fabs(dz) < z_tol_m_);
    }

    double gs() const { return std::hypot(vx_{0}, vy_{0}); }

    // horizontal acceleration estimate (m/s^2)
    double horizAccelEstimate(){
        auto nowt = this->now();
        double dt = (nowt - last_v_ts_).seconds();
        if (dt < 1e-3 || std::isnan(last_vx_) || std::isnan(last_vy_)){
            last_vx_ = vx_{0}; last_vy_ = vy_{0}; last_v_ts_ = nowt;
            return 0.0;
        }
        double ax = (vx_{0} - last_vx_) / dt;
        double ay = (vy_{0} - last_vy_) / dt;
        last_vx_ = vx_{0}; last_vy_ = vy_{0}; last_v_ts_ = nowt;
        return std::hypot(ax, ay);
    }

    void progressLog(const char* tag, const Wp& wp){
        auto nowt = this->now();
        if (last_log_ts_.nanoseconds()==0 ||
            (nowt - last_log_ts_).milliseconds() >= progress_log_ms_)
        {
            double rem = std::hypot(x_-wp.x, y_-wp.y);
            RCLCPP_INFO(get_logger(), "[%s] remain_xy=%.1fm z_err=%.1fm",
                        tag, rem, std::fabs(z_-wp.z));
            last_log_ts_ = nowt;
        }
    }

    // ---------- callbacks ----------
    void statusCb(const px4_msgs::msg::VehicleStatus::SharedPtr m){
        arming_state_ = m->arming_state;
    }
    void lposCb(const px4_msgs::msg::VehicleLocalPosition::SharedPtr m){
        if (m->xy_valid && m->z_valid){
            lpos_valid_ = true;
            x_ = m->x; y_ = m->y; z_ = m->z;
            vx_ = m->vx; vy_ = m->vy;
        } else lpos_valid_ = false;
    }
    void homeCb(const px4_msgs::msg::HomePosition::SharedPtr m){
        home_lat_ = m->latitude;
        home_lon_ = m->longitude;
        home_alt_ = m->altitude;
        home_valid_ = true;
    }

    // ---------- main control loop ----------
    void controlLoop()
    {
        switch (st_)
        {
        case St::WAIT:
            if (lpos_valid_){
                RCLCPP_INFO(get_logger(), "[WAIT] Local position valid -> ARM");
                st_ = St::ARM;
            }
            break;

        case St::ARM:
            // Send Offboard keepalive and an initial setpoint before arming
            publishOffboardKeepalive();
            publishTrajSetpoint(x_, y_, -takeoff_alt_m_, 0.0);
            if (arming_state_ != px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED){
                RCLCPP_INFO(get_logger(), "[ARM] Request arming");
                sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.f);
            } else {
                RCLCPP_INFO(get_logger(), "[ARM] Armed -> request OFFBOARD");
                sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE,
                        1.f, 6.f /* PX4_CUSTOM_MAIN_MODE_OFFBOARD */);
                st_ = St::TAKEOFF;
            }
            break;

        case St::TAKEOFF:
            publishOffboardKeepalive();
            publishTrajSetpoint(0.0, 0.0, -takeoff_alt_m_, 0.0);
            if (std::fabs(z_ + takeoff_alt_m_) < z_tol_m_){
                RCLCPP_INFO(get_logger(), "[TAKEOFF] Altitude OK -> MC_TO_WP1");
                st_ = St::MC_TO_WP1;
            }
            break;

        case St::MC_TO_WP1:
            publishOffboardKeepalive();
            publishTrajSetpoint(wp1_.x, wp1_.y, wp1_.z, wp1_.yaw_deg);
            progressLog("MC→WP1", wp1_);
            if (reachedLocal(wp1_)){
                RCLCPP_INFO(get_logger(), "[MC] WP1 reached -> CHECK_TRANSITION_COND");
                st_ = St::CHECK_TRANSITION_COND;
            }
            break;

        case St::CHECK_TRANSITION_COND:
        {
            publishOffboardKeepalive();
            publishTrajSetpoint(wp1_.x, wp1_.y, wp1_.z, wp1_.yaw_deg); // hold position
            double v = gs();
            double a_g = horizAccelEstimate() / 9.80665;
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "[TRANSITION CHECK] v=%.1f m/s |a|=%.2f g (≤%.2f g)", v, a_g, a_limit_g_);

            if (v >= v_req_ms_ && a_g <= a_limit_g_){
                RCLCPP_INFO(get_logger(), "[TRANSITION] Conditions met -> request MC->FW");
                sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_VTOL_TRANSITION, 3.f); // 3: MC->FW
                st_ = St::TRANSITION_MC2FW;
            }
            break;
        }

        case St::TRANSITION_MC2FW:
            publishOffboardKeepalive();
            RCLCPP_INFO(get_logger(), "[TRANSITION] MC->FW requested -> FW_WP2");
            st_ = St::FW_WP2;
            break;

        case St::FW_WP2:
            publishOffboardKeepalive();
            publishTrajSetpoint(wp2_.x, wp2_.y, wp2_.z, wp2_.yaw_deg);
            navToWpFw("FW→WP2", wp2_, St::FW_WP3);
            break;

        case St::FW_WP3:
            publishOffboardKeepalive();
            publishTrajSetpoint(wp3_.x, wp3_.y, wp3_.z, wp3_.yaw_deg);
            navToWpFw("FW→WP3", wp3_, St::FW_WP4);
            break;

        case St::FW_WP4:
            publishOffboardKeepalive();
            publishTrajSetpoint(wp4_.x, wp4_.y, wp4_.z, wp4_.yaw_deg);
            navToWpFw("FW→WP4", wp4_, St::FW_WP5);
            break;

        case St::FW_WP5:
            publishOffboardKeepalive();
            publishTrajSetpoint(wp5_.x, wp5_.y, wp5_.z, wp5_.yaw_deg);
            navToWpFw("FW→WP5", wp5_, St::BACK_TO_WP2);
            break;

        case St::BACK_TO_WP2:
            publishOffboardKeepalive();
            publishTrajSetpoint(wp2_.x, wp2_.y, wp2_.z, wp2_.yaw_deg);
            navToWpFw("FW back→WP2", wp2_, St::TRANSITION_FW2MC);
            break;

        case St::TRANSITION_FW2MC:
            publishOffboardKeepalive();
            RCLCPP_INFO(get_logger(), "[TRANSITION] Request FW->MC");
            sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_VTOL_TRANSITION, 4.f); // 4: FW->MC
            st_ = St::MC_TO_WP1_BACK;
            break;

        case St::MC_TO_WP1_BACK:
            publishOffboardKeepalive();
            publishTrajSetpoint(wp1_.x, wp1_.y, wp1_.z, wp1_.yaw_deg);
            progressLog("MC back→WP1", wp1_);
            if (reachedLocal(wp1_)){
                RCLCPP_INFO(get_logger(), "[MC] Passed WP1 -> LAND_HOME");
                st_ = St::LAND_HOME;
            }
            break;

        case St::LAND_HOME:
            publishOffboardKeepalive();
            if (!home_valid_){
                RCLCPP_WARN(get_logger(), "[LAND] home invalid; waiting...");
                break;
            }
            RCLCPP_INFO(get_logger(), "[LAND] NAV_LAND @HOME (global)");
            sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND,
                    0,0,0,NAN, (float)home_lat_, (float)home_lon_, (float)home_alt_);
            st_ = St::DONE;
            break;

        case St::DONE:
            publishOffboardKeepalive();
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000, "[DONE] Mission complete.");
            break;
        }
    }

    void navToWpFw(const char* tag, const Wp& wp, St next_state){
        progressLog(tag, wp);
        if (reachedLocal(wp)){
            RCLCPP_INFO(get_logger(), "[%s] reached -> next", tag);
            st_ = next_state;
        }
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VtolAutoWaypointNode>());
    rclcpp::shutdown();
    return 0;
}
