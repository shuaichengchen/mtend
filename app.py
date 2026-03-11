import math
import os
import re
import json
import time
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow, Polygon

# =========================
# 基础设置
# =========================
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei"
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = "sans-serif"

st.set_page_config(
    page_title="AI赋能中学物理虚拟实验与智能测评平台",
    page_icon="🧪",
    layout="wide"
)

DATA_FILE = "student_records.csv"
TEACHER_FILE = "teacher_accounts.json"


# =========================
# 页面美化
# =========================
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(122,162,255,0.16), transparent 22%),
            radial-gradient(circle at 90% 15%, rgba(255,166,199,0.14), transparent 22%),
            radial-gradient(circle at 20% 90%, rgba(126,220,194,0.14), transparent 20%),
            linear-gradient(135deg, #f7faff 0%, #eef4ff 45%, #f9fbff 100%);
    }

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 1350px;
}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef4ff 0%, #f8fbff 100%);
        border-right: 1px solid rgba(90,120,200,0.12);
    }

    .hero-box {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(90,120,200,0.14);
        border-radius: 28px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 45px rgba(31,42,68,0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 900;
        color: #1f2a44;
        margin-bottom: 0.3rem;
        letter-spacing: 0.3px;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #5a6887;
        line-height: 1.85;
    }

    .metric-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(90,120,200,0.12);
        border-radius: 20px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 28px rgba(31,42,68,0.07);
        min-height: 108px;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #65759d;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-size: 1.72rem;
        font-weight: 800;
        color: #1f2a44;
        line-height: 1.1;
    }

    .fancy-note {
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(245,248,255,0.96));
        border-left: 5px solid #8caeff;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        color: #394a70;
        line-height: 1.8;
        margin: 0.45rem 0 0.8rem 0;
    }

    .small-card {
        background: rgba(255,255,255,0.98);
        border: 1px solid rgba(90,120,200,0.10);
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 8px 24px rgba(31,42,68,0.05);
        margin-bottom: 0.55rem;
    }

    .small-label {
        font-size: 0.85rem;
        color: #7182aa;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .small-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #223453;
    }

    .obs-box {
        background: linear-gradient(135deg, rgba(255,252,240,0.96), rgba(255,247,225,0.96));
        border: 1px solid rgba(255,191,86,0.28);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        color: #6b4b00;
        line-height: 1.8;
        margin-top: 0.8rem;
    }

    .soft-tip {
        background: linear-gradient(135deg, rgba(238,245,255,0.96), rgba(247,250,255,0.96));
        border: 1px solid rgba(90,120,200,0.10);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        color: #34476c;
        line-height: 1.8;
        margin-top: 0.6rem;
    }

    .footer-note {
        text-align: center;
        color: #6f7fa6;
        font-size: 0.92rem;
        margin-top: 0.8rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def mini_card(label, value):
    st.markdown(
        f"""
        <div class="small-card">
            <div class="small-label">{label}</div>
            <div class="small-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(90,120,200,0.14);
            border-radius: 20px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 24px rgba(31,42,68,0.08);
            margin-bottom: 1rem;
            font-size: 1.18rem;
            font-weight: 800;
            color: #233457;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()


# =========================
# 数据读写
# =========================
def load_records():
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_record(record):
    old = load_records()
    new = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
    new.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# =========================
# 教师账号系统
# =========================
def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_teacher_accounts():
    if os.path.exists(TEACHER_FILE):
        try:
            with open(TEACHER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_teacher_accounts(accounts):
    with open(TEACHER_FILE, "w", encoding="utf-8") as f:
        json.dump(accounts, f, ensure_ascii=False, indent=2)


def register_teacher(username, password):
    accounts = load_teacher_accounts()

    if not username.strip():
        return False, "账号不能为空"
    if not password.strip():
        return False, "密码不能为空"
    if len(password) < 6:
        return False, "密码长度不能少于6位"
    if username in accounts:
        return False, "该账号已存在，请直接登录"

    accounts[username] = {
        "password": hash_password(password),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_teacher_accounts(accounts)
    return True, "注册成功，请使用该账号登录"


def login_teacher(username, password):
    accounts = load_teacher_accounts()

    if username not in accounts:
        return False, "账号不存在，请先注册"

    stored_hash = accounts[username]["password"]
    if stored_hash != hash_password(password):
        return False, "密码错误"

    return True, "登录成功"


# =========================
# Session 初始化
# =========================
if "latest_profile" not in st.session_state:
    st.session_state.latest_profile = None

if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False

if "teacher_username" not in st.session_state:
    st.session_state.teacher_username = ""

if "playing_map" not in st.session_state:
    st.session_state.playing_map = {}

if "frame_map" not in st.session_state:
    st.session_state.frame_map = {}

if "speed_map" not in st.session_state:
    st.session_state.speed_map = {}


# =========================
# 通用工具
# =========================
def get_phase_text(progress):
    if progress < 0.2:
        return "初始阶段：刚开始运动，变化趋势还不明显。"
    elif progress < 0.5:
        return "中前段：过程逐渐展开，关键量开始出现明显变化。"
    elif progress < 0.8:
        return "中后段：变化幅度增大，更适合观察规律。"
    else:
        return "结束阶段：接近终态，适合总结最终结果与规律。"


def experiment_observation(exp_name, row, progress, params):
    if exp_name == "平抛运动":
        return (
            f"当前小球已飞行 {row['时间(s)']:.2f} s，水平速度几乎保持不变，"
            f"竖直速度绝对值增大，说明平抛运动可分解为水平方向匀速运动与竖直方向自由落体运动。"
        )
    if exp_name == "自由落体":
        return (
            f"当前高度为 {row['高度y(m)']:.2f} m，速度增至 {row['速度v(m/s)']:.2f} m/s。"
            f"随着时间推移，位移增加越来越快，体现匀加速直线运动特征。"
        )
    if exp_name == "欧姆定律":
        return (
            f"此时电压为 {row['电压U(V)']:.2f} V，对应电流 {row['电流I(A)']:.2f} A。"
            f"I-U 图像上的点沿直线分布，说明电阻一定时电流与电压成正比。"
        )
    if exp_name == "凸透镜成像":
        return (
            f"此时物距 {row['物距u(cm)']:.2f} cm，像距 {row['像距v(cm)']:.2f} cm，"
            f"成像性质为“{row['像的性质']}”，像的大小为“{row['像的大小']}”。"
        )
    if exp_name == "牛顿第二定律":
        return (
            f"当前受力 {row['力F(N)']:.2f} N，加速度 {row['加速度a(m/s²)']:.2f} m/s²。"
            f"质量保持不变时，随着合外力增大，加速度同步增大。"
        )
    return (
        f"当前温度达到 {row['温度T(℃)']:.2f} ℃，总升温 {row['升温ΔT(℃)']:.2f} ℃。"
        f"在功率一定条件下，温度随时间大致稳定上升。"
    )


def draw_background_grid(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.28, color="#b7c7e3")
    for spine in ax.spines.values():
        spine.set_alpha(0.35)
        spine.set_color("#8fa6c7")


def flame_polygon(center_x, center_y, scale=1.0, phase=0.0):
    wobble = 0.08 * math.sin(phase)
    pts = [
        (center_x, center_y),
        (center_x - 0.18 * scale, center_y + 0.35 * scale),
        (center_x - 0.06 * scale, center_y + 0.75 * scale + wobble),
        (center_x, center_y + 1.10 * scale),
        (center_x + 0.08 * scale, center_y + 0.70 * scale - wobble),
        (center_x + 0.20 * scale, center_y + 0.30 * scale),
    ]
    return pts


# =========================
# 手动帧控制
# =========================
def get_manual_index(df, exp_name, title="实验过程"):
    total = len(df)
    frame_key = f"frame_{exp_name}_{title}"
    speed_key = f"speed_{exp_name}_{title}"

    if frame_key not in st.session_state:
        st.session_state[frame_key] = 0
    if speed_key not in st.session_state:
        st.session_state[speed_key] = 8

    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 3])

    play_clicked = False

    with c1:
        play_clicked = st.button("▶ 播放", key=f"play_{frame_key}")

    with c2:
        if st.button("⟲ 重置", key=f"reset_{frame_key}"):
            st.session_state[frame_key] = 0

    with c3:
        speed_options = [1, 2, 4, 8, 12, 16]
        speed = st.selectbox(
            "倍速",
            speed_options,
            index=speed_options.index(st.session_state[speed_key]),
            key=f"speed_select_{frame_key}"
        )
        st.session_state[speed_key] = speed

    with c4:
        progress = st.session_state[frame_key] / max(total - 1, 1)
        manual_progress = st.slider(
            "拖动查看实验过程",
            0.0, 1.0,
            float(progress),
            0.001,
            key=f"slider_{frame_key}"
        )

    if not play_clicked:
        st.session_state[frame_key] = min(total - 1, int(manual_progress * (total - 1)))

    return st.session_state[frame_key], play_clicked, st.session_state[speed_key]


# =========================
# 实验计算
# =========================
def simulate_projectile(v0, h, g, dt, use_drag=False, k=0.15):
    if not use_drag:
        total_time = math.sqrt(2 * h / g)
        t = np.arange(0, total_time + dt, dt)
        x = v0 * t
        y = np.maximum(h - 0.5 * g * t ** 2, 0)
        vx = np.full_like(t, v0)
        vy = -g * t
        speed = np.sqrt(vx ** 2 + vy ** 2)
    else:
        t_list, x_list, y_list = [0.0], [0.0], [h]
        vx_list, vy_list = [v0], [0.0]
        x, y, vx, vy, t = 0.0, h, v0, 0.0, 0.0
        for _ in range(100000):
            ax = -k * vx
            ay = -g - k * vy
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            t += dt
            if y < 0:
                y = 0
            t_list.append(t)
            x_list.append(x)
            y_list.append(y)
            vx_list.append(vx)
            vy_list.append(vy)
            if y <= 0:
                break
        t = np.array(t_list)
        x = np.array(x_list)
        y = np.array(y_list)
        vx = np.array(vx_list)
        vy = np.array(vy_list)
        speed = np.sqrt(vx ** 2 + vy ** 2)

    df = pd.DataFrame({
        "时间(s)": t,
        "水平位移x(m)": x,
        "竖直高度y(m)": y,
        "水平速度vx(m/s)": vx,
        "竖直速度vy(m/s)": vy,
        "合速度v(m/s)": speed
    })

    return {
        "df": df,
        "purpose": "探究平抛运动中轨迹、飞行时间与射程的变化规律。",
        "concept": "平抛运动可分解为水平方向匀速直线运动和竖直方向自由落体运动。",
        "rule": "理想情况下，飞行时间主要由高度决定，水平射程与初速度成正相关。",
        "metrics": {
            "飞行时间": f"{df['时间(s)'].iloc[-1]:.2f} s",
            "水平射程": f"{df['水平位移x(m)'].iloc[-1]:.2f} m",
            "落地速度": f"{df['合速度v(m/s)'].iloc[-1]:.2f} m/s"
        }
    }


def simulate_freefall(h, g, dt):
    total_time = math.sqrt(2 * h / g)
    t = np.arange(0, total_time + dt, dt)
    s = np.minimum(0.5 * g * t ** 2, h)
    y = np.maximum(h - s, 0)
    v = g * t
    df = pd.DataFrame({
        "时间(s)": t,
        "下落位移s(m)": s,
        "高度y(m)": y,
        "速度v(m/s)": v
    })
    return {
        "df": df,
        "purpose": "探究自由落体过程中位移和速度随时间变化的规律。",
        "concept": "自由落体是在只受重力作用时的匀加速直线运动。",
        "rule": "自由落体中速度随时间线性增大，位移与时间平方成正比。",
        "metrics": {
            "下落时间": f"{df['时间(s)'].iloc[-1]:.2f} s",
            "落地速度": f"{df['速度v(m/s)'].iloc[-1]:.2f} m/s",
            "采样点数": f"{len(df)}"
        }
    }


def simulate_ohm(voltage_max, resistance, points=20):
    u = np.linspace(0, voltage_max, points)
    i = u / resistance
    p = u * i
    df = pd.DataFrame({
        "电压U(V)": u,
        "电流I(A)": i,
        "电功率P(W)": p
    })
    return {
        "df": df,
        "purpose": "探究导体电流与电压之间的关系，理解欧姆定律。",
        "concept": "欧姆定律反映了电流、电压和电阻之间的关系。",
        "rule": "当电阻一定时，电流与电压成正比，I-U 图像是一条过原点的直线。",
        "metrics": {
            "电阻": f"{resistance:.2f} Ω",
            "最大电流": f"{i.max():.2f} A",
            "最大功率": f"{p.max():.2f} W"
        }
    }


def simulate_lens(f, u_min, u_max, step):
    u_values = np.arange(u_min, u_max + step, step)
    rows = []
    for u in u_values:
        if abs(u - f) < 1e-9:
            continue
        v = (f * u) / (u - f)
        m = abs(v / u)
        image_type = "倒立实像" if u > f else "正立虚像"
        if m > 1:
            size = "放大"
        elif abs(m - 1) < 0.05:
            size = "等大"
        else:
            size = "缩小"
        rows.append({
            "物距u(cm)": round(float(u), 3),
            "像距v(cm)": round(float(v), 3),
            "放大率": round(float(m), 3),
            "像的性质": image_type,
            "像的大小": size
        })
    df = pd.DataFrame(rows)
    return {
        "df": df,
        "purpose": "探究物距变化对凸透镜成像性质和像距变化的影响。",
        "concept": "凸透镜成像规律与物距和焦距密切相关。",
        "rule": "物体在焦点外成实像，在焦点内成正立放大的虚像。",
        "metrics": {
            "焦距": f"{f:.1f} cm",
            "记录条数": f"{len(df)}",
            "物距范围": f"{u_min:.1f}-{u_max:.1f} cm"
        }
    }


def simulate_newton2(mass, f_min, f_max, points):
    force = np.linspace(f_min, f_max, points)
    acc = force / mass
    df = pd.DataFrame({
        "力F(N)": force,
        "加速度a(m/s²)": acc
    })
    return {
        "df": df,
        "purpose": "探究质量一定时，加速度与合外力之间的关系。",
        "concept": "牛顿第二定律表明物体加速度与合外力成正比，与质量成反比。",
        "rule": "当质量一定时，加速度与合外力成正比，F-a 图像是一条过原点的直线。",
        "metrics": {
            "质量": f"{mass:.2f} kg",
            "最大合外力": f"{force.max():.2f} N",
            "最大加速度": f"{acc.max():.2f} m/s²"
        }
    }


def simulate_specific_heat(mass, c, power, total_time, dt):
    t = np.arange(0, total_time + dt, dt)
    temp_rise = power * t / (mass * c)
    temp = 20 + temp_rise
    df = pd.DataFrame({
        "时间(s)": t,
        "温度T(℃)": temp,
        "升温ΔT(℃)": temp_rise
    })
    return {
        "df": df,
        "purpose": "探究加热条件下温度变化与时间关系，理解比热容意义。",
        "concept": "比热容表示单位质量物质温度升高1℃所吸收的热量。",
        "rule": "加热功率一定时，质量越小、比热容越小，升温越快。",
        "metrics": {
            "质量": f"{mass:.2f} kg",
            "比热容": f"{c:.0f} J/(kg·℃)",
            "最终温度": f"{temp[-1]:.2f} ℃"
        }
    }


# =========================
# 图像与装置演示
# 图中统一使用英文，避免云端中文字体缺失
# =========================
def plot_projectile_trajectory(df, h, idx):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    draw_background_grid(ax, (-0.5, df["水平位移x(m)"].max() * 1.08 + 0.5), (0, max(h * 1.18, 1)))

    x = df["水平位移x(m)"].values
    y = df["竖直高度y(m)"].values

    ax.plot(x, y, color="#2c7ecb", linewidth=2.6, zorder=2)
    ax.fill_between(x[:idx+1], y[:idx+1], 0, color="#8ec5ff", alpha=0.12, zorder=1)
    ax.scatter(x[:idx+1], y[:idx+1], s=18, color="#8fc7ff", alpha=0.45, zorder=3)
    ax.scatter(x[idx], y[idx], s=125, color="#ff9a2e", edgecolor="white", linewidth=1.5, zorder=5)

    if idx < len(df) - 1:
        ax.scatter(x[-1], 0, s=90, color="#ffb56b", edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(x[-1], 0.35, "Landing", ha="center", fontsize=10, color="#9b5a00")

    vx = df["水平速度vx(m/s)"].iloc[idx]
    vy = df["竖直速度vy(m/s)"].iloc[idx]
    scale = 0.12
    ax.arrow(x[idx], y[idx], vx * scale, 0, width=0.03, head_width=0.18, head_length=0.25,
             color="#4b74d1", length_includes_head=True, zorder=6)
    ax.arrow(x[idx], y[idx], 0, vy * scale, width=0.03, head_width=0.18, head_length=0.25,
             color="#e66767", length_includes_head=True, zorder=6)

    ax.set_title("Projectile Trajectory", fontsize=16, fontweight="bold")
    ax.set_xlabel("Horizontal displacement x / m")
    ax.set_ylabel("Height y / m")
    ax.axhline(0, linestyle="--", alpha=0.35, color="#5776a2")
    fig.tight_layout()
    return fig


def draw_projectile_device(h, row, v0, idx, total):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    xmax = max(12, row["水平位移x(m)"] + 3.5)
    ymax = max(h + 1.5, 4)
    draw_background_grid(ax, (-0.5, xmax), (-0.2, ymax))

    ax.add_patch(Rectangle((0, 0), 1.45, h, color="#c9daf8", alpha=0.96, zorder=2))
    ax.add_patch(Rectangle((1.45, h - 0.32), 0.95, 0.32, color="#76a8ef", alpha=0.95, zorder=3))
    ax.plot([0, xmax], [0, 0], linewidth=2.8, color="#6ea4dc", zorder=2)

    for gx in np.arange(0, xmax, 1):
        ax.plot([gx, gx], [0, 0.13], color="#7ea8d5", alpha=0.65, linewidth=1)

    hist_x = 2.1 + row["水平位移x(m)"] * np.linspace(0.08, 1, 10)
    hist_y = np.linspace(h, row["竖直高度y(m)"], 10)
    for i, (hx, hy) in enumerate(zip(hist_x, hist_y)):
        alpha = 0.12 + i * 0.06
        ax.add_patch(Circle((hx, hy), 0.09, color="#ffb7b7", alpha=min(alpha, 0.75), zorder=4))

    ball_x = 2.1 + row["水平位移x(m)"]
    ball_y = row["竖直高度y(m)"]
    ax.add_patch(Circle((ball_x, ball_y), 0.18, color="#ff7c7c", zorder=6))
    ax.add_patch(Circle((ball_x - 0.05, ball_y + 0.05), 0.05, color="white", alpha=0.6, zorder=7))

    ax.add_patch(FancyArrow(2.0, h - 0.16, 1.0, 0, width=0.04, head_width=0.18, head_length=0.22,
                            color="#3d78c5", zorder=5))
    ax.text(2.25, h + 0.28, f"v0={v0:.1f} m/s", fontsize=11, color="#274f85")

    vy_show = row["竖直速度vy(m/s)"]
    ax.add_patch(FancyArrow(ball_x, ball_y, 0.8, 0, width=0.03, head_width=0.15, head_length=0.18,
                            color="#4c73d1", zorder=5))
    ax.add_patch(FancyArrow(ball_x, ball_y, 0, max(-0.9, vy_show * 0.05), width=0.03, head_width=0.15,
                            head_length=0.18, color="#dd6464", zorder=5))

    ax.annotate("", xy=(-0.12, h), xytext=(-0.12, 0),
                arrowprops=dict(arrowstyle="<->", color="#6a7e99", linewidth=1.6))
    ax.text(-0.35, h / 2, f"h={h:.1f}m", rotation=90, va="center", fontsize=10)

    ax.annotate("", xy=(ball_x, -0.18), xytext=(2.1, -0.18),
                arrowprops=dict(arrowstyle="<->", color="#6a7e99", linewidth=1.6))
    ax.text((ball_x + 2.1) / 2, -0.48, f"x={row['水平位移x(m)']:.2f}m", ha="center", fontsize=10)

    ax.text(ball_x + 0.22, ball_y + 0.22, "Ball", fontsize=10)
    ax.text(xmax - 2.2, ymax - 0.55, f"Frame: {idx+1}/{total}", fontsize=10, color="#607090")

    ax.set_title("Projectile Device", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_projectile_velocity(df, idx):
    fig, ax = plt.subplots(figsize=(8.8, 5))
    t = df["时间(s)"]
    ax.plot(t, df["水平速度vx(m/s)"], label="vx-t", color="#4d7ad1", linewidth=2.3)
    ax.plot(t, df["竖直速度vy(m/s)"], label="vy-t", color="#e46b6b", linewidth=2.3)
    ax.plot(t, df["合速度v(m/s)"], label="v-t", color="#ff9f3a", linewidth=2.3)
    ax.scatter(t.iloc[idx], df["合速度v(m/s)"].iloc[idx], s=95, color="#ff9f3a", edgecolor="white", zorder=5)
    ax.set_title("Velocity - Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time t / s")
    ax.set_ylabel("Velocity / (m·s⁻¹)")
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_freefall_main(df, idx):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    t = df["时间(s)"]
    y = df["高度y(m)"]
    v = df["速度v(m/s)"]

    ax.plot(t, y, color="#3f7fd0", linewidth=2.5, label="Height y-t")
    ax.scatter(t[:idx+1], y[:idx+1], color="#9eceff", s=20, alpha=0.5)
    ax.scatter(t.iloc[idx], y.iloc[idx], color="#ff9a2e", s=115, edgecolor="white", zorder=5)

    ax2 = ax.twinx()
    ax2.plot(t, v, color="#ef6d6d", linewidth=2.2, linestyle="--", label="Velocity v-t")
    ax2.scatter(t.iloc[idx], v.iloc[idx], color="#ef6d6d", s=85, edgecolor="white", zorder=5)

    ax.set_title("Free Fall: Height & Velocity", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time t / s")
    ax.set_ylabel("Height y / m")
    ax2.set_ylabel("Velocity v / (m·s⁻¹)")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def draw_freefall_device(h, row, idx, total):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    draw_background_grid(ax, (-1.6, 3.6), (-0.3, max(h + 1.1, 4)))

    ax.plot([0, 0], [0, h], linewidth=6, color="#a5bce5")
    for gy in np.arange(0, h + 0.1, max(h / 10, 1)):
        ax.plot([0, 0.18], [gy, gy], color="#6887b7", linewidth=1)
        ax.text(0.28, gy, f"{gy:.0f}", fontsize=8, va="center", color="#5e7092")

    ball_y = row["高度y(m)"]
    trail = np.linspace(h, ball_y, 10)
    for i, ty in enumerate(trail):
        ax.add_patch(Circle((0, ty), 0.09, color="#ffb5b5", alpha=0.12 + 0.06 * i, zorder=3))

    ax.add_patch(Circle((0, ball_y), 0.22, color="#ff8686", zorder=6))
    ax.add_patch(Circle((-0.04, ball_y + 0.05), 0.06, color="white", alpha=0.55, zorder=7))

    arrow_len = min(1.5, row["速度v(m/s)"] * 0.08 + 0.25)
    ax.add_patch(FancyArrow(1.0, ball_y + 0.65, 0, -arrow_len, width=0.05,
                            head_width=0.18, head_length=0.18, color="#4e7bd1"))
    ax.text(1.18, ball_y + 0.2, "Velocity", fontsize=10)

    ax.annotate("", xy=(-0.7, h), xytext=(-0.7, 0),
                arrowprops=dict(arrowstyle="<->", color="#6a7e99", linewidth=1.6))
    ax.text(-1.0, h / 2, f"h={h:.1f}m", rotation=90, va="center", fontsize=10)
    ax.text(1.7, h + 0.25, f"Frame: {idx+1}/{total}", fontsize=10, color="#607090")

    ax.set_title("Free Fall Device", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_ohm(df, idx):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    u = df["电压U(V)"]
    i = df["电流I(A)"]
    ax.plot(u, i, marker="o", color="#3f7fd0", linewidth=2.4)
    ax.fill_between(u[:idx+1], i[:idx+1], color="#8ec5ff", alpha=0.13)
    ax.scatter(u.iloc[idx], i.iloc[idx], s=120, color="#ff9b2f", edgecolor="white", zorder=5)
    ax.set_title("Ohm's Law: I-U Curve", fontsize=16, fontweight="bold")
    ax.set_xlabel("Voltage U / V")
    ax.set_ylabel("Current I / A")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def draw_ohm_device(row, resistance, idx, total):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    draw_background_grid(ax, (0, 7.3), (0.2, 4.2))

    ax.plot([0.5, 2.0], [2, 2], linewidth=3.2, color="#2f4058")
    ax.plot([2.0, 2.6], [2, 2], linewidth=3.2, color="#2f4058")
    ax.plot([2.6, 3.0], [2.25, 1.75], linewidth=3.2, color="#2f4058")
    ax.plot([3.0, 3.4], [1.75, 2.25], linewidth=3.2, color="#2f4058")
    ax.plot([3.4, 3.8], [2.25, 1.75], linewidth=3.2, color="#2f4058")
    ax.plot([3.8, 4.4], [1.75, 2], linewidth=3.2, color="#2f4058")
    ax.plot([4.4, 6.0], [2, 2], linewidth=3.2, color="#2f4058")
    ax.plot([6.0, 6.0], [2, 0.9], linewidth=3.2, color="#2f4058")
    ax.plot([6.0, 0.5], [0.9, 0.9], linewidth=3.2, color="#2f4058")
    ax.plot([0.5, 0.5], [0.9, 2], linewidth=3.2, color="#2f4058")

    ax.add_patch(Circle((1.2, 2), 0.26, fill=False, linewidth=2.2, edgecolor="#5b79ab"))
    ax.text(1.11, 1.89, "A", fontsize=12, color="#334c77", fontweight="bold")

    ax.add_patch(Circle((3.2, 3.0), 0.28, fill=False, linewidth=2.2, edgecolor="#5b79ab"))
    ax.text(3.1, 2.88, "V", fontsize=12, color="#334c77", fontweight="bold")
    ax.plot([2.3, 2.3], [2, 3], linewidth=2.1, color="#2f4058")
    ax.plot([4.1, 4.1], [2, 3], linewidth=2.1, color="#2f4058")
    ax.plot([2.3, 3.2], [3, 3], linewidth=2.1, color="#2f4058")
    ax.plot([4.1, 3.2], [3, 3], linewidth=2.1, color="#2f4058")

    ax.add_patch(Rectangle((5.0, 1.55), 0.65, 0.95, fill=False, linewidth=2.2, edgecolor="#d56e6e"))
    ax.text(5.08, 2.65, "Power", fontsize=10, color="#8d3f3f")

    dots = np.linspace(0.75, 5.8, 8)
    for i, _ in enumerate(dots):
        offset = (idx * 0.1 + i * 0.3) % 5.0
        x = 0.75 + offset
        if x <= 6.0:
            ax.add_patch(Circle((x, 2.0), 0.05, color="#55c2ff", alpha=0.78))

    ax.text(2.72, 1.18, f"R={resistance:.1f}Ω", fontsize=11, color="#475d84")
    ax.text(0.78, 2.58, f"I={row['电流I(A)']:.2f}A", fontsize=11, color="#475d84")
    ax.text(2.82, 3.37, f"U={row['电压U(V)']:.2f}V", fontsize=11, color="#475d84")
    ax.text(5.7, 3.55, f"Frame: {idx+1}/{total}", fontsize=10, color="#607090")

    ax.set_title("Circuit Diagram", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_lens(df, f, idx):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    u = df["物距u(cm)"]
    v = df["像距v(cm)"]
    ax.plot(u, v, marker="o", color="#3f7fd0", linewidth=2.4)
    ax.axvline(f, linestyle="--", alpha=0.55, color="#e58a4f", label="Focus f")
    ax.scatter(u.iloc[idx], v.iloc[idx], s=125, color="#ff9b2f", edgecolor="white", zorder=5)
    ax.set_title("Lens: Object Distance vs Image Distance", fontsize=16, fontweight="bold")
    ax.set_xlabel("Object distance u / cm")
    ax.set_ylabel("Image distance v / cm")
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    return fig


def draw_lens_device(f, row, idx, total):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    draw_background_grid(ax, (-10, 10), (-3.2, 3.2))

    ax.plot([-10, 10], [0, 0], linewidth=2.2, color="#2f4058")
    ax.axvline(0, linewidth=3.4, color="#5a8ddb")
    ax.axvline(-f, linestyle="--", alpha=0.45, color="#e2a86b")
    ax.axvline(f, linestyle="--", alpha=0.45, color="#e2a86b")
    ax.axvline(-2 * f, linestyle=":", alpha=0.35, color="#b4bfd8")
    ax.axvline(2 * f, linestyle=":", alpha=0.35, color="#b4bfd8")
    ax.text(-f, -0.4, "F", ha="center", fontsize=11)
    ax.text(f, -0.4, "F", ha="center", fontsize=11)
    ax.text(-2 * f, -0.4, "2F", ha="center", fontsize=10)
    ax.text(2 * f, -0.4, "2F", ha="center", fontsize=10)

    u = row["物距u(cm)"] / 4
    v = np.clip(row["像距v(cm)"] / 4, -9.2, 9.2)
    image_type = row["像的性质"]
    image_size = row["像的大小"]

    obj_x = -u
    obj_h = 1.45
    ax.add_patch(FancyArrow(obj_x, 0, 0, obj_h, width=0.08, head_width=0.28, head_length=0.18,
                            color="#ff7f7f", zorder=5))
    ax.text(obj_x - 0.3, obj_h + 0.18, "Object", fontsize=10)

    img_h = 1.2 if image_size == "缩小" else (1.9 if image_size == "放大" else 1.45)
    if image_type == "倒立实像":
        ax.add_patch(FancyArrow(v, 0, 0, -img_h, width=0.08, head_width=0.28, head_length=0.18,
                                color="#4a7bd1", zorder=5))
        ax.text(v - 0.28, -img_h - 0.35, "Image", fontsize=10)
    else:
        ax.add_patch(FancyArrow(v, 0, 0, img_h, width=0.08, head_width=0.28, head_length=0.18,
                                color="#4a7bd1", zorder=5))
        ax.text(v - 0.28, img_h + 0.15, "Image", fontsize=10)

    if image_type == "倒立实像":
        ax.plot([obj_x, 0, v], [obj_h, obj_h, 0], color="#f2a444", linewidth=1.6)
        ax.plot([obj_x, 0, v], [obj_h, 0, -img_h], color="#7ab0ff", linewidth=1.6)
        ax.plot([obj_x, v], [obj_h, -img_h], color="#8ad0b5", linewidth=1.6)
    else:
        ax.plot([obj_x, 0, 3.5], [obj_h, obj_h, obj_h], color="#f2a444", linewidth=1.6)
        ax.plot([obj_x, 0, 3.8], [obj_h, 0, -0.8], color="#7ab0ff", linewidth=1.6)
        ax.plot([obj_x, 3.2], [obj_h, 2.6], color="#8ad0b5", linewidth=1.6)

    ax.text(-9.3, 2.35, f"u={row['物距u(cm)']:.1f} cm", fontsize=10, color="#475d84")
    ax.text(3.8, 2.35, f"v={row['像距v(cm)']:.1f} cm", fontsize=10, color="#475d84")
    ax.text(0.7, 2.35, f"{image_type} / {image_size}", fontsize=10, color="#475d84")
    ax.text(7.0, 2.75, f"Frame: {idx+1}/{total}", fontsize=10, color="#607090")

    ax.set_title("Lens Imaging Device", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_newton(df, idx):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    f = df["力F(N)"]
    a = df["加速度a(m/s²)"]
    ax.plot(f, a, marker="o", color="#3f7fd0", linewidth=2.4)
    ax.fill_between(f[:idx+1], a[:idx+1], color="#8ec5ff", alpha=0.13)
    ax.scatter(f.iloc[idx], a.iloc[idx], s=120, color="#ff9b2f", edgecolor="white", zorder=5)
    ax.set_title("Newton's Second Law: F-a Curve", fontsize=16, fontweight="bold")
    ax.set_xlabel("Force F / N")
    ax.set_ylabel("Acceleration a / (m/s²)")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def draw_newton_device(row, mass, idx, total):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    draw_background_grid(ax, (0, 11), (0, 3.4))
    ax.plot([0, 10.5], [0.8, 0.8], linewidth=4, color="#9fb8ea")
    for x in np.arange(0.5, 10.5, 1):
        ax.plot([x, x], [0.8, 0.95], color="#7b98cb", linewidth=1)

    cart_x = 2.2 + min(5.5, row["力F(N)"] * 0.16)
    ax.add_patch(Rectangle((cart_x, 0.8), 1.7, 0.95, color="#8cb4f2", alpha=0.95))
    ax.add_patch(Rectangle((cart_x + 0.15, 1.15), 1.35, 0.3, color="#b8d1fa", alpha=0.8))
    ax.add_patch(Circle((cart_x + 0.35, 0.72), 0.12, color="#344057"))
    ax.add_patch(Circle((cart_x + 1.35, 0.72), 0.12, color="#344057"))

    arrow_len = min(3.2, row["力F(N)"] / 3 + 0.5)
    ax.add_patch(FancyArrow(cart_x + 1.95, 1.28, arrow_len, 0, width=0.06, head_width=0.2,
                            head_length=0.25, color="#e85d5d"))
    ax.add_patch(FancyArrow(cart_x - 0.15, 1.02, -0.6, 0, width=0.03, head_width=0.13,
                            head_length=0.12, color="#8aa0bf", alpha=0.5))
    ax.text(cart_x + 2.2, 1.58, f"F={row['力F(N)']:.2f}N", fontsize=11, color="#8a4141")
    ax.text(cart_x + 0.2, 2.02, f"m={mass:.2f}kg", fontsize=11, color="#475d84")
    ax.text(cart_x + 0.2, 2.34, f"a={row['加速度a(m/s²)']:.2f}m/s²", fontsize=11, color="#475d84")
    ax.text(8.2, 2.8, f"Frame: {idx+1}/{total}", fontsize=10, color="#607090")

    ax.set_title("Newton Device", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_heat(df, idx):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    t = df["时间(s)"]
    temp = df["温度T(℃)"]
    ax.plot(t, temp, color="#ef6d6d", linewidth=2.6)
    ax.fill_between(t[:idx+1], temp[:idx+1], color="#ffc0b2", alpha=0.15)
    ax.scatter(t.iloc[idx], temp.iloc[idx], s=120, color="#ff9b2f", edgecolor="white", zorder=5)
    ax.set_title("Temperature - Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time t / s")
    ax.set_ylabel("Temperature T / ℃")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def draw_heat_device(row, power, mass, idx, total):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    draw_background_grid(ax, (0, 8.5), (0, 3.4))

    ax.add_patch(Rectangle((3.15, 0.55), 2.35, 1.95, fill=False, linewidth=2.2, edgecolor="#6f87ab"))
    water_height = min(1.55, 0.72 + (row["温度T(℃)"] - 20) / 100)
    ax.add_patch(Rectangle((3.2, 0.55), 2.25, water_height, color="#8fd3ff", alpha=0.88))

    xs = np.linspace(3.2, 5.45, 80)
    ys = 0.55 + water_height + 0.03 * np.sin((xs - 3.2) * 10 + idx * 0.45)
    ax.plot(xs, ys, color="#5ab7e9", linewidth=1.2)

    ax.plot([6.4, 6.4], [0.7, 2.6], color="#7c8fb0", linewidth=3)
    temp_ratio = min(1.0, (row["温度T(℃)"] - 20) / max(1, 120 - 20))
    mercury_h = 0.7 + 1.7 * temp_ratio
    ax.plot([6.4, 6.4], [0.7, mercury_h], color="#ef6d6d", linewidth=6)
    ax.add_patch(Circle((6.4, 0.58), 0.12, color="#ef6d6d"))

    ax.add_patch(Rectangle((4.15, 0.0), 0.28, 0.62, color="#ff8b5c"))
    ax.add_patch(Polygon(flame_polygon(4.29, 0.62, 0.55, phase=idx), closed=True,
                         color="#ff9f3a", alpha=0.95))
    ax.add_patch(Polygon(flame_polygon(4.29, 0.74, 0.32, phase=idx + 1.5), closed=True,
                         color="#ffe07c", alpha=0.95))

    ax.text(0.8, 2.5, f"P={power:.0f}W", fontsize=11, color="#475d84")
    ax.text(0.8, 2.05, f"m={mass:.2f}kg", fontsize=11, color="#475d84")
    ax.text(0.8, 1.6, f"T={row['温度T(℃)']:.2f}℃", fontsize=11, color="#475d84")
    ax.text(6.95, 2.8, f"Frame: {idx+1}/{total}", fontsize=10, color="#607090")

    ax.set_title("Heating Device", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_profile(concept, image_score, rule_score, process_score):
    labels = ["Concept", "Image", "Rule", "Process"]
    values = [concept, image_score, rule_score, process_score]
    colors = ["#7faef5", "#8fd3c8", "#ffc46b", "#f59aa0"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_title("Student Profile", fontsize=16, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    return fig


def plot_growth_curve(df_student):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = list(range(1, len(df_student) + 1))
    ax.plot(x, df_student["综合得分"], marker="o", linewidth=2.5, color="#3f7fd0")
    ax.fill_between(x, df_student["综合得分"], alpha=0.10, color="#8ec5ff")
    ax.set_title("Growth Curve", fontsize=16, fontweight="bold")
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Total score")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


# =========================
# 题库
# =========================
QUESTION_BANK = {
    "平抛运动": {
        "concept": [
            {"question": "1. 平抛运动水平方向的运动性质是？", "options": ["匀速直线运动", "匀加速直线运动", "静止"], "answer": "匀速直线运动", "analysis": "水平方向不受力，做匀速直线运动。"},
            {"question": "2. 平抛运动竖直方向的运动性质是？", "options": ["自由落体运动", "匀速直线运动", "静止"], "answer": "自由落体运动", "analysis": "竖直方向只受重力作用。"}
        ],
        "image": [
            {"question": "3. 理想平抛的轨迹形状是？", "options": ["抛物线", "圆", "直线"], "answer": "抛物线", "analysis": "分运动合成后是抛物线。"},
            {"question": "4. 理想平抛中的 vx 随时间如何变化？", "options": ["保持不变", "逐渐增大", "逐渐减小"], "answer": "保持不变", "analysis": "理想平抛水平方向无加速度。"}
        ],
        "rule": [
            {"question": "5. 忽略空气阻力时，飞行时间主要由什么决定？", "options": ["抛出高度", "水平初速度", "物体质量"], "answer": "抛出高度", "analysis": "由竖直方向运动决定。"},
            {"question": "6. 在高度不变时，水平初速度增大，射程将？", "options": ["变大", "变小", "不变"], "answer": "变大", "analysis": "飞行时间近似不变，速度越大射程越大。"}
        ]
    },
    "自由落体": {
        "concept": [
            {"question": "1. 自由落体运动中物体主要受什么力？", "options": ["重力", "摩擦力", "弹力"], "answer": "重力", "analysis": "理想条件下只受重力。"},
            {"question": "2. 自由落体属于哪类运动？", "options": ["匀加速直线运动", "匀速直线运动", "曲线运动"], "answer": "匀加速直线运动", "analysis": "其加速度近似恒定为 g。"}
        ],
        "image": [
            {"question": "3. 自由落体的 v-t 图像通常是？", "options": ["过原点的直线", "水平直线", "抛物线"], "answer": "过原点的直线", "analysis": "速度与时间成正比。"},
            {"question": "4. 自由落体的 s-t 图像通常更接近？", "options": ["抛物线", "直线", "圆弧"], "answer": "抛物线", "analysis": "位移与时间平方成正比。"}
        ],
        "rule": [
            {"question": "5. 自由落体过程中，速度会怎样变化？", "options": ["均匀增大", "保持不变", "均匀减小"], "answer": "均匀增大", "analysis": "加速度恒定，速度均匀增大。"},
            {"question": "6. 地面附近重力加速度通常近似为？", "options": ["9.8 m/s²", "1 m/s²", "98 m/s²"], "answer": "9.8 m/s²", "analysis": "常用近似值。"}
        ]
    },
    "欧姆定律": {
        "concept": [
            {"question": "1. 欧姆定律表达式是？", "options": ["I = U / R", "U = I / R", "R = U × I"], "answer": "I = U / R", "analysis": "电流等于电压除以电阻。"},
            {"question": "2. 当电阻一定时，电压增大，电流会？", "options": ["增大", "减小", "不变"], "answer": "增大", "analysis": "电流与电压成正比。"}
        ],
        "image": [
            {"question": "3. 纯电阻导体的 I-U 图像通常是？", "options": ["过原点的直线", "水平线", "抛物线"], "answer": "过原点的直线", "analysis": "说明 I 与 U 成正比。"},
            {"question": "4. I-U 图线越陡，说明电阻通常？", "options": ["越小", "越大", "不变"], "answer": "越小", "analysis": "斜率约为 1/R。"}
        ],
        "rule": [
            {"question": "5. 电阻不变时，U 与 I 的关系是？", "options": ["成正比", "成反比", "无关"], "answer": "成正比", "analysis": "欧姆定律基本结论。"},
            {"question": "6. 若 U=6V，R=3Ω，则 I=? ", "options": ["2A", "3A", "18A"], "answer": "2A", "analysis": "I = U/R = 2A。"}
        ]
    },
    "凸透镜成像": {
        "concept": [
            {"question": "1. 凸透镜对平行光的作用是？", "options": ["会聚", "发散", "不变"], "answer": "会聚", "analysis": "凸透镜能会聚平行光。"},
            {"question": "2. 物体位于焦点内时通常成？", "options": ["正立虚像", "倒立实像", "等大实像"], "answer": "正立虚像", "analysis": "焦内成正立放大虚像。"}
        ],
        "image": [
            {"question": "3. 物距接近焦点外侧时，像距通常会？", "options": ["增大", "减小", "不变"], "answer": "增大", "analysis": "靠近焦点外侧时，像距变大。"},
            {"question": "4. 物距大于两倍焦距时，像通常是？", "options": ["缩小", "放大", "不变"], "answer": "缩小", "analysis": "此时成倒立缩小实像。"}
        ],
        "rule": [
            {"question": "5. 凸透镜成实像时，物体应位于？", "options": ["焦点外", "焦点内", "焦点上"], "answer": "焦点外", "analysis": "焦点外才可成实像。"},
            {"question": "6. 照相机应用的是哪类成像？", "options": ["倒立缩小实像", "正立放大虚像", "等大实像"], "answer": "倒立缩小实像", "analysis": "照相机底片上成倒立缩小实像。"}
        ]
    },
    "牛顿第二定律": {
        "concept": [
            {"question": "1. 牛顿第二定律可表示为？", "options": ["F = ma", "F = mv", "P = W/t"], "answer": "F = ma", "analysis": "合外力等于质量与加速度的乘积。"},
            {"question": "2. 在合外力一定时，质量越大，加速度会？", "options": ["越小", "越大", "不变"], "answer": "越小", "analysis": "a = F/m。"}
        ],
        "image": [
            {"question": "3. 质量一定时，a-F 图像一般是？", "options": ["过原点的直线", "水平线", "抛物线"], "answer": "过原点的直线", "analysis": "说明加速度与力成正比。"},
            {"question": "4. 斜率越大，表示物体质量通常？", "options": ["越小", "越大", "不变"], "answer": "越小", "analysis": "a/F = 1/m。"}
        ],
        "rule": [
            {"question": "5. 质量一定时，加速度与合外力关系为？", "options": ["成正比", "成反比", "无关"], "answer": "成正比", "analysis": "质量一定时 a ∝ F。"},
            {"question": "6. 若 m=2kg，F=6N，则 a=? ", "options": ["3m/s²", "12m/s²", "8m/s²"], "answer": "3m/s²", "analysis": "a = F/m = 3。"}
        ]
    },
    "比热容与升温": {
        "concept": [
            {"question": "1. 比热容表示什么？", "options": ["单位质量物质升高1℃所吸收的热量", "单位时间内做的功", "单位面积所受压力"], "answer": "单位质量物质升高1℃所吸收的热量", "analysis": "这是比热容定义。"},
            {"question": "2. 在相同加热条件下，比热容越大，升温会？", "options": ["越慢", "越快", "不变"], "answer": "越慢", "analysis": "比热容越大，同样热量带来的温升越小。"}
        ],
        "image": [
            {"question": "3. 恒定功率加热时，T-t 图像通常接近？", "options": ["上升直线", "水平线", "下降曲线"], "answer": "上升直线", "analysis": "简化条件下温度近似随时间线性变化。"},
            {"question": "4. 质量更小的物体，升温曲线通常？", "options": ["更陡", "更平", "相同"], "answer": "更陡", "analysis": "ΔT = Pt/(mc)。"}
        ],
        "rule": [
            {"question": "5. 在功率一定时，升温快慢与哪两个量有关？", "options": ["质量和比热容", "电阻和电压", "速度和位移"], "answer": "质量和比热容", "analysis": "由 ΔT = Pt/(mc) 可知。"},
            {"question": "6. 当质量和比热容都增大时，升温速度会？", "options": ["减慢", "加快", "不变"], "answer": "减慢", "analysis": "分母增大，温升速率下降。"}
        ]
    }
}


# =========================
# 文本分析
# =========================
SUMMARY_KEYWORDS = {
    "平抛运动": ["匀速", "自由落体", "抛物线", "高度", "射程", "水平", "竖直"],
    "自由落体": ["重力", "匀加速", "速度", "位移", "时间平方", "加速度"],
    "欧姆定律": ["电压", "电流", "电阻", "正比", "I=U/R", "图像", "直线"],
    "凸透镜成像": ["焦点", "实像", "虚像", "物距", "像距", "放大", "缩小"],
    "牛顿第二定律": ["力", "加速度", "质量", "F=ma", "正比", "反比"],
    "比热容与升温": ["比热容", "热量", "温度", "质量", "升温", "功率", "吸热"]
}


def analyze_summary_text(text, exp_name):
    text = str(text).strip()
    keywords = SUMMARY_KEYWORDS.get(exp_name, [])
    if not text:
        return {
            "score": 0,
            "level": "未作答",
            "feedback": ["你还没有填写规律总结，建议至少写出实验现象、变量关系和结论。"]
        }

    text_len = len(text)
    hit_keywords = sum(1 for kw in keywords if kw in text)
    has_relation = any(x in text for x in ["成正比", "成反比", "随", "增大", "减小", "无关", "有关"])
    has_structure = any(x in text for x in ["因为", "所以", "说明", "由此", "可见", "因此"])
    has_symbols = bool(re.search(r"[=∝<>]", text))

    length_score = min(25, int(text_len / 3))
    keyword_score = int(hit_keywords / max(len(keywords), 1) * 45)
    relation_score = 15 if has_relation else 0
    structure_score = 10 if has_structure else 0
    symbol_score = 5 if has_symbols else 0

    total = min(100, length_score + keyword_score + relation_score + structure_score + symbol_score)

    if total >= 85:
        level = "优秀"
    elif total >= 70:
        level = "良好"
    elif total >= 50:
        level = "中等"
    else:
        level = "待提升"

    feedback = []
    if hit_keywords < max(2, len(keywords) // 3):
        feedback.append("关键词覆盖较少，建议补充核心物理术语。")
    if not has_relation:
        feedback.append("总结中缺少变量关系描述，例如“成正比”“增大”“减小”等。")
    if not has_structure:
        feedback.append("建议用“现象—分析—结论”的方式书写，更像规范实验报告。")
    if text_len < 30:
        feedback.append("文字略少，可再补充实验条件和规律适用范围。")
    if not feedback:
        feedback.append("你的规律总结较完整，已经具备较好的实验表述能力。")

    return {
        "score": total,
        "level": level,
        "feedback": feedback
    }


# =========================
# 自动评语
# =========================
def generate_comment_and_advice(concept_score, image_score, rule_score, process_score, exp_name):
    overall = np.mean([concept_score, image_score, rule_score, process_score])

    if overall >= 85:
        comment = f"该生在“{exp_name}”实验中表现优秀，能够较好地理解概念、分析图像并总结规律，实验参与度也较高。"
    elif overall >= 70:
        comment = f"该生在“{exp_name}”实验中整体表现较好，已具备较好的基础理解能力，但部分维度仍可继续加强。"
    elif overall >= 55:
        comment = f"该生在“{exp_name}”实验中达到基本要求，但在概念理解、图像分析或规律归纳方面仍存在提升空间。"
    else:
        comment = f"该生在“{exp_name}”实验中的掌握程度偏弱，建议进一步进行基础知识回顾与实验训练。"

    weak_points = []
    advice = []

    if concept_score < 60:
        weak_points.append("概念理解较弱")
        advice.append("建议回顾实验原理和核心公式，再重新观察实验过程。")
    if image_score < 60:
        weak_points.append("图像分析较弱")
        advice.append("建议多结合图像描述变量变化趋势，训练读图与析图能力。")
    if rule_score < 60:
        weak_points.append("规律总结较弱")
        advice.append("建议尝试用“条件—现象—结论”的方式写实验总结。")
    if process_score < 60:
        weak_points.append("过程参与度不足")
        advice.append("建议多进行参数调节、图像查看与实验反思，提高探究参与度。")

    if not weak_points:
        weak_points.append("暂无明显薄弱项")
        advice.append("建议继续挑战更多实验并进行跨实验比较分析。")

    return comment, weak_points, advice


# =========================
# 评分
# =========================
def score_quiz(exp_name, answers):
    bank = QUESTION_BANK[exp_name]
    scores = {"concept": 0, "image": 0, "rule": 0}
    feedback = []

    name_map = {
        "concept": "概念理解",
        "image": "图像总结分析",
        "rule": "规律总结分析"
    }

    for category in ["concept", "image", "rule"]:
        questions = bank[category]
        for idx, q in enumerate(questions):
            key = f"{category}_{idx}"
            if answers.get(key) == q["answer"]:
                scores[category] += 1
            else:
                feedback.append(
                    f"{name_map[category]}：{q['question']}；正确答案：{q['answer']}；提示：{q['analysis']}"
                )

    concept_score = int(scores["concept"] / len(bank["concept"]) * 100)
    image_score = int(scores["image"] / len(bank["image"]) * 100)
    rule_score = int(scores["rule"] / len(bank["rule"]) * 100)

    return concept_score, image_score, rule_score, feedback


# =========================
# 实验演示统一渲染
# =========================
def render_state_panel(exp_name, row, params, result, progress):
    st.markdown(f'<div class="fancy-note">{result["purpose"]}</div>', unsafe_allow_html=True)
    st.info(result["concept"])
    st.success(result["rule"])

    st.markdown("### 实时状态")
    c1, c2 = st.columns(2)

    with c1:
        if exp_name == "平抛运动":
            mini_card("当前时刻", f"{row['时间(s)']:.2f} s")
            mini_card("水平位移", f"{row['水平位移x(m)']:.2f} m")
            mini_card("当前高度", f"{row['竖直高度y(m)']:.2f} m")
        elif exp_name == "自由落体":
            mini_card("当前时刻", f"{row['时间(s)']:.2f} s")
            mini_card("下落位移", f"{row['下落位移s(m)']:.2f} m")
            mini_card("当前高度", f"{row['高度y(m)']:.2f} m")
        elif exp_name == "欧姆定律":
            mini_card("当前电压", f"{row['电压U(V)']:.2f} V")
            mini_card("当前电流", f"{row['电流I(A)']:.2f} A")
            mini_card("当前功率", f"{row['电功率P(W)']:.2f} W")
        elif exp_name == "凸透镜成像":
            mini_card("当前物距", f"{row['物距u(cm)']:.2f} cm")
            mini_card("当前像距", f"{row['像距v(cm)']:.2f} cm")
            mini_card("像的性质", f"{row['像的性质']}")
        elif exp_name == "牛顿第二定律":
            mini_card("当前合外力", f"{row['力F(N)']:.2f} N")
            mini_card("当前加速度", f"{row['加速度a(m/s²)']:.2f} m/s²")
            mini_card("质量保持", f"{params['mass']:.2f} kg")
        else:
            mini_card("当前时刻", f"{row['时间(s)']:.2f} s")
            mini_card("当前温度", f"{row['温度T(℃)']:.2f} ℃")
            mini_card("总升温", f"{row['升温ΔT(℃)']:.2f} ℃")

    with c2:
        if exp_name == "平抛运动":
            mini_card("水平速度", f"{row['水平速度vx(m/s)']:.2f} m/s")
            mini_card("竖直速度", f"{row['竖直速度vy(m/s)']:.2f} m/s")
            mini_card("合速度", f"{row['合速度v(m/s)']:.2f} m/s")
        elif exp_name == "自由落体":
            mini_card("当前速度", f"{row['速度v(m/s)']:.2f} m/s")
            mini_card("重力加速度", f"{params['g']:.1f} m/s²")
            mini_card("阶段提示", "加速下落")
        elif exp_name == "欧姆定律":
            mini_card("电阻", f"{params['resistance']:.2f} Ω")
            mini_card("电流关系", "I ∝ U")
            mini_card("图像特征", "过原点直线")
        elif exp_name == "凸透镜成像":
            mini_card("像的大小", f"{row['像的大小']}")
            mini_card("焦距", f"{params['f']:.1f} cm")
            mini_card("成像类别", "实像/虚像变化")
        elif exp_name == "牛顿第二定律":
            mini_card("规律关系", "a ∝ F")
            mini_card("质量", f"{params['mass']:.2f} kg")
            mini_card("图像特征", "过原点直线")
        else:
            mini_card("加热功率", f"{params['power']:.0f} W")
            mini_card("质量", f"{params['mass']:.2f} kg")
            mini_card("比热容", f"{params['c']:.0f} J/(kg·℃)")

    st.markdown(
        f"""
        <div class="obs-box">
        <b>观察提示：</b>{experiment_observation(exp_name, row, progress, params)}<br>
        <b>过程阶段：</b>{get_phase_text(progress)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="soft-tip">
        建议课堂使用方式：先播放动画观察实验过程，再暂停在关键时刻分析图像，最后完成规律总结与测试，
        形成“观察—分析—总结—反馈”的学习闭环。
        </div>
        """,
        unsafe_allow_html=True
    )


def render_experiment_demo(experiment_name, df, params, result):
    total = len(df)
    idx, play_clicked, speed = get_manual_index(df, experiment_name)

    chart_placeholder = st.empty()

    def draw_frame(frame_idx):
        row = df.iloc[frame_idx]
        progress = frame_idx / max(total - 1, 1)

        with chart_placeholder.container():
left_col, right_col = st.columns([1, 1])

with left_col:
    if experiment_name == "平抛运动":
        st.pyplot(plot_projectile_trajectory(df, params["h"], frame_idx), use_container_width=True, clear_figure=True)
    elif experiment_name == "自由落体":
        st.pyplot(plot_freefall_main(df, frame_idx), use_container_width=True, clear_figure=True)
    elif experiment_name == "欧姆定律":
        st.pyplot(plot_ohm(df, frame_idx), use_container_width=True, clear_figure=True)
    elif experiment_name == "凸透镜成像":
        st.pyplot(plot_lens(df, params["f"], frame_idx), use_container_width=True, clear_figure=True)
    elif experiment_name == "牛顿第二定律":
        st.pyplot(plot_newton(df, frame_idx), use_container_width=True, clear_figure=True)
    else:
        st.pyplot(plot_heat(df, frame_idx), use_container_width=True, clear_figure=True)

with right_col:
    if experiment_name == "平抛运动":
        st.pyplot(draw_projectile_device(params["h"], row, params["v0"], frame_idx, len(df)), use_container_width=True, clear_figure=True)
    elif experiment_name == "自由落体":
        st.pyplot(draw_freefall_device(params["h"], row, frame_idx, len(df)), use_container_width=True, clear_figure=True)
    elif experiment_name == "欧姆定律":
        st.pyplot(draw_ohm_device(row, params["resistance"], frame_idx, len(df)), use_container_width=True, clear_figure=True)
    elif experiment_name == "凸透镜成像":
        st.pyplot(draw_lens_device(params["f"] / 4, row, frame_idx, len(df)), use_container_width=True, clear_figure=True)
    elif experiment_name == "牛顿第二定律":
        st.pyplot(draw_newton_device(row, params["mass"], frame_idx, len(df)), use_container_width=True, clear_figure=True)
    else:
        st.pyplot(draw_heat_device(row, params["power"], params["mass"], frame_idx, len(df)), use_container_width=True, clear_figure=True)

st.markdown("### 实时状态与观察")
render_state_panel(experiment_name, row, params, result, progress)

st.markdown("### 引导思考")
if experiment_name == "平抛运动":
    st.write("1. 为什么水平方向速度几乎不变？")
    st.write("2. 为什么竖直方向速度越来越大？")
    st.write("3. 为什么轨迹不是直线而是抛物线？")
elif experiment_name == "自由落体":
    st.write("1. 为什么下落越到后面越快？")
    st.write("2. 为什么速度—时间图像接近直线？")
    st.write("3. 为什么位移与时间平方有关？")
elif experiment_name == "欧姆定律":
    st.write("1. 为什么电压越大，电流越大？")
    st.write("2. 为什么图像是一条直线？")
    st.write("3. 电阻改变后图像斜率会怎样变化？")
elif experiment_name == "凸透镜成像":
    st.write("1. 物体靠近焦点时像距为什么会明显变化？")
    st.write("2. 为什么焦内会形成虚像？")
    st.write("3. 物距变化时像的正倒和大小如何变化？")
elif experiment_name == "牛顿第二定律":
    st.write("1. 为什么质量不变时 a 会随 F 增大而增大？")
    st.write("2. 图像过原点说明了什么？")
    st.write("3. 如果质量变大，图像会发生什么变化？")
else:
    st.write("1. 为什么加热时间越长温度越高？")
    st.write("2. 为什么质量或比热容变大时升温会变慢？")
    st.write("3. 恒定功率加热时温度变化趋势有何特点？")

        return row

    current_row = draw_frame(idx)

    if play_clicked:
        step = max(1, int(speed))
        for frame_idx in range(idx, total, step):
            st.session_state[f"frame_{experiment_name}_实验过程"] = frame_idx
            current_row = draw_frame(frame_idx)
            time.sleep(0.08)

        st.session_state[f"frame_{experiment_name}_实验过程"] = total - 1

    return current_row


def render_analysis_tab(experiment_name, df, params):
    idx = len(df) - 1 if len(df) > 0 else 0

    if experiment_name == "平抛运动":
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_projectile_velocity(df, idx), use_container_width=True, clear_figure=True)
        with c2:
            fig, ax = plt.subplots(figsize=(8.8, 5))
            ax.plot(df["时间(s)"], df["水平位移x(m)"], label="x-t", color="#3f7fd0", linewidth=2.3)
            ax.plot(df["时间(s)"], df["竖直高度y(m)"], label="y-t", color="#ef6d6d", linewidth=2.3)
            ax.set_title("Displacement - Time", fontsize=16, fontweight="bold")
            ax.set_xlabel("Time t / s")
            ax.grid(True, alpha=0.22)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True, clear_figure=True)
        st.write("图像结论：理想平抛中，vx 基本保持不变，vy 随时间线性变化，轨迹为抛物线。")

    elif experiment_name == "自由落体":
        fig, ax = plt.subplots(figsize=(8.8, 5))
        ax.plot(df["时间(s)"], df["下落位移s(m)"], label="s-t", color="#3f7fd0", linewidth=2.4)
        ax.plot(df["时间(s)"], df["速度v(m/s)"], label="v-t", color="#ef6d6d", linewidth=2.4)
        ax.set_title("Free Fall Curves", fontsize=16, fontweight="bold")
        ax.set_xlabel("Time t / s")
        ax.grid(True, alpha=0.22)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True, clear_figure=True)
        st.write("图像结论：v-t 图像接近过原点直线，s-t 图像体现位移与时间平方关系。")

    elif experiment_name == "欧姆定律":
        st.pyplot(plot_ohm(df, idx), use_container_width=True, clear_figure=True)
        st.write("图像结论：I-U 图像是一条过原点的直线，说明电阻一定时电流与电压成正比。")

    elif experiment_name == "凸透镜成像":
        st.pyplot(plot_lens(df, params["f"], idx), use_container_width=True, clear_figure=True)
        st.write("图像结论：物距接近焦点外侧时，像距会快速增大；在焦内可形成虚像。")

    elif experiment_name == "牛顿第二定律":
        st.pyplot(plot_newton(df, idx), use_container_width=True, clear_figure=True)
        st.write("图像结论：质量一定时，a-F 图像近似过原点直线，说明加速度与力成正比。")

    else:
        st.pyplot(plot_heat(df, idx), use_container_width=True, clear_figure=True)
        st.write("图像结论：恒定功率加热时，温度随时间近似线性升高。")


# =========================
# 侧边栏
# =========================
st.sidebar.markdown("## 平台设置")
role = st.sidebar.radio("选择使用模式", ["学生端", "教师端"])


# =========================
# 学生端准备数据
# =========================
if role == "学生端":
    st.sidebar.markdown("### 学生信息")
    student_id = st.sidebar.text_input("学号", value="2026001")
    class_name = st.sidebar.text_input("班级", value="高一(1)班")
    student_name = st.sidebar.text_input("姓名", value="学生A")

    st.sidebar.markdown("### 实验模块")
    experiment_name = st.sidebar.selectbox(
        "选择实验",
        ["平抛运动", "自由落体", "欧姆定律", "凸透镜成像", "牛顿第二定律", "比热容与升温"]
    )

    show_data = st.sidebar.checkbox("显示数据表", value=True)
    show_chart = st.sidebar.checkbox("显示图像分析", value=True)

    params = {}
    defaults_changed = False

    if experiment_name == "平抛运动":
        use_drag = st.sidebar.radio("模型类型", ["理想模型", "空气阻力模型"]) == "空气阻力模型"
        params["v0"] = st.sidebar.slider("水平初速度 v₀（m/s）", 1.0, 30.0, 10.0, 0.5)
        params["h"] = st.sidebar.slider("抛出高度 h（m）", 0.5, 50.0, 10.0, 0.5)
        params["g"] = st.sidebar.slider("重力加速度 g（m/s²）", 1.0, 20.0, 9.8, 0.1)
        params["dt"] = st.sidebar.slider("时间步长 dt（s）", 0.005, 0.1, 0.03, 0.005)
        params["use_drag"] = use_drag
        if use_drag:
            params["k"] = st.sidebar.slider("空气阻力系数 k", 0.01, 1.0, 0.15, 0.01)
        defaults_changed = params["v0"] != 10.0 or params["h"] != 10.0 or use_drag
        result = simulate_projectile(**params)
        df = result["df"]

    elif experiment_name == "自由落体":
        params["h"] = st.sidebar.slider("下落高度 h（m）", 1.0, 100.0, 20.0, 1.0)
        params["g"] = st.sidebar.slider("重力加速度 g（m/s²）", 1.0, 20.0, 9.8, 0.1)
        params["dt"] = st.sidebar.slider("时间步长 dt（s）", 0.005, 0.1, 0.03, 0.005)
        defaults_changed = params["h"] != 20.0
        result = simulate_freefall(**params)
        df = result["df"]

    elif experiment_name == "欧姆定律":
        params["voltage_max"] = st.sidebar.slider("最大电压 Umax（V）", 1.0, 24.0, 12.0, 0.5)
        params["resistance"] = st.sidebar.slider("电阻 R（Ω）", 1.0, 20.0, 5.0, 0.5)
        params["points"] = st.sidebar.slider("采样点数", 5, 50, 20, 1)
        defaults_changed = params["resistance"] != 5.0
        result = simulate_ohm(**params)
        df = result["df"]

    elif experiment_name == "凸透镜成像":
        params["f"] = st.sidebar.slider("焦距 f（cm）", 5.0, 30.0, 10.0, 0.5)
        params["u_min"] = st.sidebar.slider("最小物距 u_min（cm）", 2.0, 20.0, 5.0, 0.5)
        params["u_max"] = st.sidebar.slider("最大物距 u_max（cm）", 15.0, 100.0, 40.0, 1.0)
        params["step"] = st.sidebar.slider("物距步长（cm）", 0.5, 10.0, 2.0, 0.5)
        defaults_changed = params["f"] != 10.0
        result = simulate_lens(**params)
        df = result["df"]

    elif experiment_name == "牛顿第二定律":
        params["mass"] = st.sidebar.slider("质量 m（kg）", 0.5, 10.0, 2.0, 0.1)
        params["f_min"] = st.sidebar.slider("最小力 Fmin（N）", 0.0, 10.0, 0.0, 0.5)
        params["f_max"] = st.sidebar.slider("最大力 Fmax（N）", 1.0, 50.0, 20.0, 0.5)
        params["points"] = st.sidebar.slider("采样点数", 5, 40, 15, 1)
        defaults_changed = params["mass"] != 2.0
        result = simulate_newton2(**params)
        df = result["df"]

    else:
        params["mass"] = st.sidebar.slider("质量 m（kg）", 0.1, 5.0, 1.0, 0.1)
        params["c"] = st.sidebar.slider("比热容 c（J/kg·℃）", 100.0, 5000.0, 4200.0, 50.0)
        params["power"] = st.sidebar.slider("加热功率 P（W）", 50.0, 2000.0, 500.0, 10.0)
        params["total_time"] = st.sidebar.slider("总加热时间（s）", 10.0, 600.0, 120.0, 10.0)
        params["dt"] = st.sidebar.slider("时间步长（s）", 1.0, 20.0, 5.0, 1.0)
        defaults_changed = params["mass"] != 1.0 or params["c"] != 4200.0
        result = simulate_specific_heat(**params)
        df = result["df"]


# =========================
# 学生端页面
# =========================
if role == "学生端":
    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-title">AI赋能中学物理虚拟实验与智能测评平台</div>
            <div class="hero-subtitle">
                当前学生：<b>{student_name}</b>（{class_name}）&nbsp;&nbsp;|&nbsp;&nbsp;
                当前实验：<b>{experiment_name}</b><br>
                平台支持多实验仿真、动画演示、图像分析、规律总结、实验测试、过程性评价、智能评语与教师端统计分析。
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    metric_items = list(result["metrics"].items())
    with c1:
        metric_card(metric_items[0][0], metric_items[0][1])
    with c2:
        metric_card(metric_items[1][0], metric_items[1][1])
    with c3:
        metric_card(metric_items[2][0], metric_items[2][1])
    with c4:
        metric_card("实验类别", experiment_name)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["实验演示", "图像分析", "规律总结", "实验测试", "学习画像", "平台说明"]
    )

    with tab1:
        section_title(f"{experiment_name}实验演示")
        row = render_experiment_demo(experiment_name, df, params, result)

        if show_data:
            st.markdown("### 实验数据表")
            st.dataframe(df, use_container_width=True, height=300)
            st.download_button(
                "下载实验数据 CSV",
                to_csv_bytes(df),
                file_name=f"{experiment_name}_实验数据.csv",
                mime="text/csv"
            )

    with tab2:
        section_title(f"{experiment_name}图像分析")
        if show_chart:
            render_analysis_tab(experiment_name, df, params)
        else:
            st.warning("你在侧边栏关闭了图像分析显示。")

    with tab3:
        section_title(f"{experiment_name}规律总结与文本分析")
        st.markdown("### 示例规律")
        st.markdown(
            f"""
            <div class="fancy-note">
            {result["rule"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        student_summary = st.text_area(
            "请写出你对本实验规律的总结",
            height=170,
            placeholder="建议写出：实验现象、变量关系、结论以及适用条件……"
        )

        summary_analysis = analyze_summary_text(student_summary, experiment_name)

        a1, a2 = st.columns(2)
        with a1:
            st.metric("规律总结文本分", f"{summary_analysis['score']} / 100")
        with a2:
            st.metric("文本质量等级", summary_analysis["level"])

        for item in summary_analysis["feedback"]:
            st.write(f"- {item}")

    with tab4:
        section_title(f"{experiment_name}实验测试")
        st.markdown("请完成以下 6 道题，系统会从概念理解、图像分析、规律总结三个维度自动评分。")

        bank = QUESTION_BANK[experiment_name]
        answers = {}

        for category in ["concept", "image", "rule"]:
            title_map = {
                "concept": "概念理解题",
                "image": "图像总结分析题",
                "rule": "规律总结分析题"
            }
            st.markdown(f"### {title_map[category]}")
            for idx, q in enumerate(bank[category]):
                answers[f"{category}_{idx}"] = st.radio(
                    q["question"],
                    q["options"],
                    key=f"{experiment_name}_{category}_{idx}"
                )

        if st.button("提交测试并生成智能评价"):
            concept_score, image_score, rule_quiz_score, feedback = score_quiz(experiment_name, answers)

            summary_score = summary_analysis["score"]

            process_score = 0
            if defaults_changed:
                process_score += 25
            if show_chart:
                process_score += 25
            if len(student_summary.strip()) >= 20:
                process_score += 25
            process_score += 25

            rule_score = int(0.6 * rule_quiz_score + 0.4 * summary_score)

            total_score = int(
                concept_score * 0.25 +
                image_score * 0.25 +
                rule_score * 0.30 +
                process_score * 0.20
            )

            comment, weak_points, advice = generate_comment_and_advice(
                concept_score, image_score, rule_score, process_score, experiment_name
            )

            record = {
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "学号": student_id,
                "班级": class_name,
                "姓名": student_name,
                "实验名称": experiment_name,
                "概念理解分": concept_score,
                "图像分析分": image_score,
                "规律总结分": rule_score,
                "过程参与分": process_score,
                "综合得分": total_score,
                "自动评语": comment
            }
            save_record(record)
            st.session_state.latest_profile = record

            d1, d2, d3, d4, d5 = st.columns(5)
            with d1:
                st.metric("概念理解分", f"{concept_score}")
            with d2:
                st.metric("图像分析分", f"{image_score}")
            with d3:
                st.metric("规律总结分", f"{rule_score}")
            with d4:
                st.metric("过程参与分", f"{process_score}")
            with d5:
                st.metric("综合得分", f"{total_score}")

            st.markdown("### 自动评语")
            st.info(comment)

            st.markdown("### 薄弱点诊断")
            for item in weak_points:
                st.write(f"- {item}")

            st.markdown("### 个性化建议")
            for item in advice:
                st.write(f"- {item}")

            st.markdown("### 错题反馈")
            if feedback:
                for item in feedback:
                    st.write(f"- {item}")
            else:
                st.success("客观题全部答对，说明你对本实验基础内容掌握较好。")

            st.success("本次成绩已经保存到本地 CSV 文件，可在教师端查看班级统计结果。")

    with tab5:
        section_title("学生能力画像与成长记录")
        profile = st.session_state.latest_profile
        all_records = load_records()

        if profile is not None:
            st.markdown(f"### 当前学生：{profile['姓名']} | 最近实验：{profile['实验名称']}")
            st.pyplot(
                plot_profile(
                    profile["概念理解分"],
                    profile["图像分析分"],
                    profile["规律总结分"],
                    profile["过程参与分"]
                ),
                use_container_width=True,
                clear_figure=True
            )
            st.write(profile["自动评语"])
        else:
            st.info("请先在“实验测试”页面提交一次测试，系统会生成学生能力画像。")

        if not all_records.empty:
            student_records = all_records[all_records["姓名"] == student_name].copy()
            if not student_records.empty:
                st.markdown("### 个人成长曲线")
                st.pyplot(plot_growth_curve(student_records), use_container_width=True, clear_figure=True)

                st.markdown("### 个人历史记录")
                st.dataframe(student_records, use_container_width=True, height=250)
            else:
                st.warning("当前学生暂无历史记录。")

            st.download_button(
                "下载全部学生记录 CSV",
                to_csv_bytes(all_records),
                file_name="student_records.csv",
                mime="text/csv"
            )

    with tab6:
        section_title("平台说明")
        st.markdown("""
### 已实现功能
1. 多个中学物理实验：平抛运动、自由落体、欧姆定律、凸透镜成像、牛顿第二定律、比热容与升温  
2. 提供动画演示、实验装置拟物示意图、参数调节、动态图像观察、数据表和规律总结  
3. 实验测试与多维评价：概念理解、图像分析、规律总结、过程参与  
4. 学生学习画像、成长曲线与历史记录  
5. 教师端账号注册登录与教学统计分析  

### 本版重点升级
- 六个实验都加入了更精美的实验过程展示  
- 加入播放、重置、倍速控制和滑块查看  
- 加入轨迹拖尾、动态高亮、拟物装置、速度/位移箭头  
- 图内文字改为英文，减少云端字体兼容问题  

### 后续可继续扩展
- 增加更多实验模块  
- 加入数据库和多用户管理  
- 接入更强的自然语言处理模型  
- 自动生成实验报告与 PDF 导出  
        """)


# =========================
# 教师端
# =========================
else:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">教师端数据分析与教学诊断</div>
            <div class="hero-subtitle">
                教师登录后可查看班级整体实验表现、实验分布、平均得分、薄弱维度和个体成长情况，
                用于课堂教学反思与学情诊断。
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.teacher_logged_in:
        section_title("教师登录 / 注册")

        login_tab, register_tab = st.tabs(["教师登录", "教师注册"])

        with login_tab:
            login_username = st.text_input("教师账号", key="login_username")
            login_password = st.text_input("教师密码", type="password", key="login_password")

            if st.button("登录教师端"):
                success, msg = login_teacher(login_username, login_password)
                if success:
                    st.session_state.teacher_logged_in = True
                    st.session_state.teacher_username = login_username
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with register_tab:
            reg_username = st.text_input("注册账号", key="reg_username")
            reg_password = st.text_input("注册密码", type="password", key="reg_password")
            reg_password2 = st.text_input("确认密码", type="password", key="reg_password2")

            if st.button("注册教师账号"):
                if reg_password != reg_password2:
                    st.error("两次输入的密码不一致")
                else:
                    success, msg = register_teacher(reg_username, reg_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown(
            """
            <div class="soft-tip">
            当前版本使用本地 JSON 文件保存教师账号，密码以哈希形式存储，适合课程设计与毕业设计演示。
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.success(f"当前已登录教师账号：{st.session_state.teacher_username}")
        with col_b:
            if st.button("退出登录"):
                st.session_state.teacher_logged_in = False
                st.session_state.teacher_username = ""
                st.rerun()

        records = load_records()

        if records.empty:
            st.warning("当前还没有学生提交记录。请先在学生端完成一次测试。")
        else:
            class_filter = st.selectbox(
                "筛选班级",
                ["全部班级"] + sorted(records["班级"].dropna().astype(str).unique().tolist())
            )
            exp_filter = st.selectbox(
                "筛选实验",
                ["全部实验"] + sorted(records["实验名称"].dropna().astype(str).unique().tolist())
            )

            filtered = records.copy()
            if class_filter != "全部班级":
                filtered = filtered[filtered["班级"] == class_filter]
            if exp_filter != "全部实验":
                filtered = filtered[filtered["实验名称"] == exp_filter]

            avg_concept = filtered["概念理解分"].mean() if not filtered.empty else 0
            avg_image = filtered["图像分析分"].mean() if not filtered.empty else 0
            avg_rule = filtered["规律总结分"].mean() if not filtered.empty else 0
            avg_process = filtered["过程参与分"].mean() if not filtered.empty else 0
            avg_total = filtered["综合得分"].mean() if not filtered.empty else 0

            t1, t2, t3, t4, t5 = st.columns(5)
            with t1:
                metric_card("平均概念分", f"{avg_concept:.1f}")
            with t2:
                metric_card("平均图像分", f"{avg_image:.1f}")
            with t3:
                metric_card("平均规律分", f"{avg_rule:.1f}")
            with t4:
                metric_card("平均过程分", f"{avg_process:.1f}")
            with t5:
                metric_card("平均综合分", f"{avg_total:.1f}")

            st.markdown("### 班级薄弱点诊断")
            weak_summary = {
                "概念理解": avg_concept,
                "图像分析": avg_image,
                "规律总结": avg_rule,
                "过程参与": avg_process
            }
            weak_sorted = sorted(weak_summary.items(), key=lambda x: x[1])
            for name, score in weak_sorted:
                st.write(f"- {name}：{score:.1f} 分")

            st.markdown("### 各实验平均综合得分")
            exp_stats = filtered.groupby("实验名称", as_index=False)["综合得分"].mean()
            st.dataframe(exp_stats, use_container_width=True, height=220)

            if not exp_stats.empty:
                fig, ax = plt.subplots(figsize=(8.8, 4.8))
                colors = ["#7faef5", "#8fd3c8", "#ffc46b", "#f59aa0", "#a7b8ff", "#8ec5ff"]
                ax.bar(exp_stats["实验名称"], exp_stats["综合得分"], color=colors[:len(exp_stats)])
                ax.set_ylim(0, 100)
                ax.set_title("Average Score by Experiment", fontsize=16, fontweight="bold")
                ax.set_ylabel("Average score")
                ax.grid(axis="y", alpha=0.22)
                plt.xticks(rotation=20)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True, clear_figure=True)

            st.markdown("### 学生个人记录查询")
            names = sorted(filtered["姓名"].dropna().astype(str).unique().tolist())
            if names:
                selected_student = st.selectbox("选择学生", names)

                stu_df = filtered[filtered["姓名"] == selected_student].copy()
                if not stu_df.empty:
                    st.dataframe(stu_df, use_container_width=True, height=240)
                    st.pyplot(plot_growth_curve(stu_df), use_container_width=True, clear_figure=True)
                    latest = stu_df.iloc[-1]
                    st.info(f"最近一次评语：{latest['自动评语']}")

            st.markdown("### 完整数据表")
            st.dataframe(filtered, use_container_width=True, height=320)

            st.download_button(
                "下载筛选后的教师统计数据 CSV",
                to_csv_bytes(filtered),
                file_name="教师端统计数据.csv",
                mime="text/csv"
            )

st.markdown(
    '<div class="footer-note">开发说明：本平台使用 Python + Streamlit + NumPy + Pandas + Matplotlib 实现，已包含多实验、动画演示、学生端、教师端、账号注册登录、过程性评价与文本分析。</div>',
    unsafe_allow_html=True
)
