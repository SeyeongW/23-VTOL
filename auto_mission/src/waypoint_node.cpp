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
        v_req_ms_            = this->declare_parameter<double>("forward_req_speed_ms", 12.0);   // 천이 전 속도 조건 (m/s)
        a_limit_g_           = this->declare_parameter<double>("transition_acc_g_max", 0.3);   // 천이시 |a| ≤ 0.3 g
        xy_tol_m_            = this->declare_parameter<double>("xy_tol_m",             6.0);
        z_tol_m_             = this->declare_parameter<double>("z_tol_m",              3.0);
        ctrl_period_ms_      = this->declare_parameter<int>("control_period_ms",       100);
        progress_log_ms_     = this->declare_parameter<int>("progress_log_ms",         1500);

        // ---------------- I/O ----------------
        status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
            "/fmu/out/vehicle_status", 10,
            std::bind(&VtolAutoWaypointNode::statusCb, this, _1));

        lpos_sub_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            "/fmu/out/vehicle_local_position", 10,
            std::bind(&VtolAutoWaypointNode::lposCb, this, _1));

        home_sub_ = this->create_subscription<px4_msgs::msg::HomePosition>(
            "/fmu/out/home_position", 10,
            std::bind(&VtolAutoWaypointNode::homeCb, this, _1));

        cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(ctrl_period_ms_),
            std::bind(&VtolAutoWaypointNode::controlLoop, this));

        RCLCPP_INFO(get_logger(), "VTOL auto mission node (WP1 MC -> WP2 transition -> WP3-5 FW -> back WP2 transition -> WP1 -> HOME LAND)");

        // ---------------- Waypoints (LOCAL NED) ----------------
        // 필요하면 GLOBAL도 혼용 가능하지만, 여기선 NED 고정 예시로 간단히.
        // z는 Down(+). -takeoff_alt_m_가 고도 takeoff_alt_m_ 의미.
        wp1_ = {0,  60.0,   0.0, -takeoff_alt_m_,  0.0f}; // MC로 먼저 갈 지점
        wp2_ = {0, 100.0,  30.0, -takeoff_alt_m_, 45.0f}; // 천이 후 첫 지점
        wp3_ = {0, 140.0,   0.0, -takeoff_alt_m_,  0.0f};
        wp4_ = {0, 180.0, -30.0, -takeoff_alt_m_, -45.0f};
        wp5_ = {0, 220.0,   0.0, -takeoff_alt_m_,  0.0f};

        // 가속도 추정을 위한 초기화
        last_vx_ = NAN; last_vy_ = NAN; last_v_ts_ = this->now();
    }

private:
    // ---------- Types ----------
    struct Wp {
        int frame;        // 0: LOCAL(NED)
        double x, y, z;   // local meters (N,E,Down)
        float yaw_deg;
    };

    enum class St {
        WAIT = 0,
        ARM, TAKEOFF,
        MC_TO_WP1,
        CHECK_TRANSITION_COND,   // (v >= v_req) && (|a_horiz| <= 0.3 g)
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

    // accel estimate
    double last_vx_{NAN}, last_vy_{NAN};
    rclcpp::Time last_v_ts_;

    // ---------- mission ----------
    Wp wp1_, wp2_, wp3_, wp4_, wp5_;
    St st_{St::WAIT};
    rclcpp::Time last_log_ts_;

    // ---------- params ----------
    double takeoff_alt_m_{20.0};
    double v_req_ms_{12.0};
    double a_limit_g_{0.3};
    double xy_tol_m_{6.0};
    double z_tol_m_{3.0};
    int    ctrl_period_ms_{100};
    int    progress_log_ms_{1500};

    // ---------- ROS ----------
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr lpos_sub_;
    rclcpp::Subscription<px4_msgs::msg::HomePosition>::SharedPtr home_sub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr cmd_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ---------- helpers ----------
    uint64_t nowUsec() const { return this->get_clock()->now().nanoseconds()/1000; }
    double gs() const { return std::hypot(vx_{0}, vy_{0}); }

    void sendCmd(uint16_t cmd, float p1=0,float p2=0,float p3=0,float p4=0,float p5=0,float p6=0,float p7=0){
        px4_msgs::msg::VehicleCommand c{};
        c.timestamp = nowUsec();
        c.param1=p1; c.param2=p2; c.param3=p3; c.param4=p4; c.param5=p5; c.param6=p6; c.param7=p7;
        c.command = cmd;
        c.target_system=1; c.target_component=1; c.source_system=1; c.source_component=1;
        c.from_external = true;
        cmd_pub_->publish(c);
    }

    bool reachedLocal(const Wp &wp) const {
        double dx=x_-wp.x, dy=y_-wp.y, dz=z_-wp.z;
        return (std::hypot(dx,dy) < xy_tol_m_) && (std::fabs(dz) < z_tol_m_);
    }

    void gotoLocalAsGlobal(const Wp &wp){ // 홈 기준 ENU~LLA 변환 생략: SITL 데모에선 NAV_WAYPOINT 글로벌 대신 로컬 Progress만 사용
        // 실제 비행에선 local->global 변환하여 NAV_WAYPOINT로 지시하는 게 안전.
        // 여기서는 단순히 주기적으로 진행상황만 로깅하며, 도달 판정은 local로 처리.
        (void)wp; // no-op command here; PX4 navigator에 실제 global waypoint를 넣고 싶으면 변환/전송 코드를 추가.
    }

    // 수평가속도 추정 (m/s^2) – vx,vy의 차분
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

    // ---------- callbacks ----------
    void statusCb(const px4_msgs::msg::VehicleStatus::SharedPtr m){
        arming_state_ = m->arming_state;
    }
    void lposCb(const px4_msgs::msg::VehicleLocalPosition::SharedPtr m){
        if (m->xy_valid && m->z_valid){
            lpos_valid_ = true;
            x_ = m->x; y_ = m->y; z_ = m->z;
            vx_ = m->vx; vy_ = m->vy;
        }else lpos_valid_ = false;
    }
    void homeCb(const px4_msgs::msg::HomePosition::SharedPtr m){
        // 주의: PX4 버전에 따라 단위가 다를 수 있음. (여기선 SITL 기본 가정)
        home_lat_ = m->latitude;
        home_lon_ = m->longitude;
        home_alt_ = m->altitude;
        home_valid_ = true;
    }

    // ---------- main loop ----------
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
            if (arming_state_ != px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED){
                RCLCPP_INFO(get_logger(), "[ARM] Arm request");
                sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.f);
            } else {
                RCLCPP_INFO(get_logger(), "[ARM] Armed -> TAKEOFF %.1fm", takeoff_alt_m_);
                sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_TAKEOFF, 0,0,0,NAN, 0,0, (float)takeoff_alt_m_);
                st_ = St::TAKEOFF;
            }
            break;

        case St::TAKEOFF:
            if (std::fabs(z_ + takeoff_alt_m_) < z_tol_m_){
                RCLCPP_INFO(get_logger(), "[TAKEOFF] Alt ok -> MC_TO_WP1");
                st_ = St::MC_TO_WP1;
            }
            break;

        case St::MC_TO_WP1:
            progressLog("MC→WP1", wp1_);
            // (실제 제어명령 없이 PX4 navigator/position controller가 움직인다고 가정한 데모)
            gotoLocalAsGlobal(wp1_);
            if (reachedLocal(wp1_)){
                RCLCPP_INFO(get_logger(), "[MC] WP1 reached -> CHECK_TRANSITION_COND");
                st_ = St::CHECK_TRANSITION_COND;
            }
            break;

        case St::CHECK_TRANSITION_COND:
        {
            double v = gs();
            double a = horizAccelEstimate();          // m/s^2
            double a_g = a / 9.80665;                 // g 단위
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "[TRANSITION CHECK] v=%.1f m/s  |a|=%.2f g (limit=%.2f g)",
                                 v, a_g, a_limit_g_);

            if (v >= v_req_ms_ && a_g <= a_limit_g_){
                RCLCPP_INFO(get_logger(), "[TRANSITION] Conditions met -> request MC->FW");
                sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_VTOL_TRANSITION, 3.f); // 3: MC->FW
                st_ = St::TRANSITION_MC2FW;
            }
            break;
        }

        case St::TRANSITION_MC2FW:
            RCLCPP_INFO(get_logger(), "[TRANSITION] MC->FW requested -> FW_WP2");
            st_ = St::FW_WP2;
            break;

        case St::FW_WP2:
            navToWpFw("FW→WP2", wp2_, St::FW_WP3);
            break;
        case St::FW_WP3:
            navToWpFw("FW→WP3", wp3_, St::FW_WP4);
            break;
        case St::FW_WP4:
            navToWpFw("FW→WP4", wp4_, St::FW_WP5);
            break;
        case St::FW_WP5:
            navToWpFw("FW→WP5", wp5_, St::BACK_TO_WP2);
            break;

        case St::BACK_TO_WP2:
            navToWpFw("FW back→WP2", wp2_, St::TRANSITION_FW2MC);
            break;

        case St::TRANSITION_FW2MC:
            RCLCPP_INFO(get_logger(), "[TRANSITION] Request FW->MC");
            sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_VTOL_TRANSITION, 4.f); // 4: FW->MC
            st_ = St::MC_TO_WP1_BACK;
            break;

        case St::MC_TO_WP1_BACK:
            progressLog("MC back→WP1", wp1_);
            if (reachedLocal(wp1_)){
                RCLCPP_INFO(get_logger(), "[MC] Passed WP1 -> LAND_HOME");
                st_ = St::LAND_HOME;
            }
            break;

        case St::LAND_HOME:
            if (!home_valid_){
                RCLCPP_WARN(get_logger(), "[LAND] home invalid; waiting...");
                break;
            }
            RCLCPP_INFO(get_logger(), "[LAND] NAV_LAND @HOME");
            sendCmd(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND,
                    0,0,0,NAN, (float)home_lat_, (float)home_lon_, (float)home_alt_);
            st_ = St::DONE;
            break;

        case St::DONE:
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000, "[DONE] Mission complete.");
            break;
        }
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

    void navToWpFw(const char* tag, const Wp& wp, St next_state){
        progressLog(tag, wp);
        // (필요시 여기서 local->global 변환 후 NAV_WAYPOINT 전송 코드를 추가 가능)
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
