# HEVEN Hackathon 2026 — 팀 워크스페이스

MIT / MORAI 시뮬레이터용 ROS 2 워크스페이스입니다.

```
heven_hackathon_2026_ws/
├── mit_ws/     # MIT 시뮬레이터용 패키지 (src/hk_mit)
├── morai_ws/   # MORAI 시뮬레이터용 패키지 (src/hk_morai)
└── data/       # 시뮬레이터 출력 (Git 제외)
```

---

## ⚠️ 중요: 폴더 이름은 반드시 `heven_hackathon_2026_ws`

Docker 컨테이너는 `~/heven_hackathon_2026_ws` 경로를 자동으로 마운트합니다.
**저장소 이름(`skku_hackerton`) 그대로 clone하면 컨테이너와 연결되지 않습니다.**
아래 명령처럼 clone할 때 꼭 폴더 이름을 `heven_hackathon_2026_ws`로 지정하세요.

기존 `heven_hackathon_2026_ws` 폴더가 있으면 `heven_hackathon_2026_ws_backup`으로 이름을 바꿔 백업한 뒤 clone합니다.

---

## 1. 저장소 받기 (Clone)

### Windows (PowerShell)

```powershell
cd $HOME
if (Test-Path heven_hackathon_2026_ws) { Rename-Item heven_hackathon_2026_ws heven_hackathon_2026_ws_backup }
git clone https://github.com/kimeunsol68-lang/skku_hackerton.git heven_hackathon_2026_ws
```

### macOS / Ubuntu (터미널)

```bash
cd ~
[ -d heven_hackathon_2026_ws ] && mv heven_hackathon_2026_ws heven_hackathon_2026_ws_backup
git clone https://github.com/kimeunsol68-lang/skku_hackerton.git heven_hackathon_2026_ws
```

> 백업 폴더(`heven_hackathon_2026_ws_backup`)가 이미 있으면 이름 변경이 실패합니다. 먼저 지우거나 다른 이름으로 바꾸세요.

---

## 2. 코드 업데이트 후 빌드 (필수)

`git pull`로 최신 코드를 받은 뒤에는 **컨테이너 안에서 다시 빌드해야** 변경 사항이 반영됩니다.

```bash
# 호스트에서
cd ~/heven_hackathon_2026_ws        # Windows: cd $HOME\heven_hackathon_2026_ws
git pull

# 컨테이너 내부에서
cd /mit_ws
colcon build && source install/setup.bash
```

빌드 후 실행 예시:

```bash
ros2 launch hk_mit bringup.launch.py
```

> 새 터미널을 열 때마다 `source install/setup.bash`를 다시 실행하세요.

---

## 3. 팀원별 담당 파일 (MIT: `mit_ws/src/hk_mit/`)

> 🚨 **자기 담당 파일만 수정하세요.** 다른 사람 파일을 고치면 merge 때 충돌(Conflict)이 납니다.
> 공용 파일이나 남의 파일을 바꿔야 하면, 직접 고치지 말고 그 파일 담당자에게 요청하세요.

| 담당 | 파일 (`hk_mit/` 기준) | 내용 |
|---|---|---|
| **기본 주행·제어** | `control.py`, `planner.py`, `config/bringup.yaml` | 고속 주행, 곡률 기반 감속, 조향/구동 지연 보상, 저마찰·경사 제어 |
| **팀원 1** (Localization & Sensor) | `localization.py`, `lidar_perception.py` | GNSS/IMU 필터, 음영 구간 추측항법, LiDAR 노이즈 필터 + 전처리 토픽 발행 |
| **팀원 2** (주차) | `mission_parking.py` | CP8→CP9 평행주차, CP9→CP10 후면주차 |
| **추후 선착순** (장애물) | `mission_obstacle.py` | CP1→CP2 장애물 1개, CP6→CP7 장애물 3개 회피 |
| **추후 선착순** (로터리) | `mission_rotary.py` | CP5→CP6 로터리 NPC 감지 + 진입 타이밍 |
| 공용 (바꿀 때 팀 전체 상의) | `mission_base.py`, `mission_manager.py`, `cp_manager.py`, `course.py`, `sim_events.py`, `utm.py`, `setup.py`, `launch/` | 미션 인터페이스, 미션 스위치, CP 구간 판별 등 |

### 동작 구조

```
localization ──/hk/local/odom──▶ planner ──/hk/plan/local_path──▶ control ──/hk/control/command──▶ 시뮬레이터
lidar_perception ──/hk/perception/obstacles──▶ │   (미션 override_path가 있으면 그 경로)    ▲
                                               └──/hk/mission/override_cmd────────────────┘
                                                  (미션 override_cmd가 있으면 Stanley/PI 대신 이 명령)
```

- `planner.py`는 20 Hz마다 기본 경로를 만든 뒤 `cp_manager`로 **지금 CP 구간**을 판별합니다. 그 구간을 맡은 미션의 `update()`만 호출합니다.
- 미션이 모두 `None`을 돌려주면 **원래 기본 주행(전역 경로 + Stanley/PI)과 완전히 똑같이** 동작합니다.
- 미션 코드에서 예외가 나도 주행은 멈추지 않습니다. 로그만 남기고 기본 주행으로 처리됩니다.
- `config/bringup.yaml`에서 `missions_enabled: false`로 두면 미션 스위치를 통째로 끌 수 있습니다.

### 미션 파일 입출력 규격

각 미션 파일의 클래스는 `MissionBase`를 상속하고 아래 세 메서드를 채웁니다.

```python
class ParkingMission(MissionBase):
    NAME = 'parking'
    SECTIONS = (('cp8', 'cp9'), ('cp9', 'cp10'))   # 이 구간에서만 update()가 불림

    def reset(self): ...                 # 시작·체크포인트 복귀 때
    def on_enter(self, section): ...     # 담당 구간에 들어올 때 한 번
    def update(self, inp: MissionInput) -> MissionOutput: ...   # 20 Hz
```

**입력 `inp: MissionInput`**: map 좌표계, SI 단위

| 필드 | 타입 | 설명 |
|---|---|---|
| `x, y, yaw, speed` | float | 차 위치(m, 뒤차축)·방향(rad)·속력(m/s) |
| `local_path` | `[(x, y, yaw), ...]` | 기본 경로 (차 뒤 2 m ~ 앞 30 m, 0.5 m 간격) |
| `scan` | `LaserScan` \| None | 원본 LiDAR (`lidar_link` 좌표) |
| `obstacles` | `[(x, y), ...]` \| None | 팀원 1의 전처리 결과 (map 좌표) |
| `t` | float | 시뮬레이션 시간 (s) |
| `cp` | `CPState` | `cp.section` 예: `('cp8','cp9')`, `cp.dist_to_next_m` = 다음 게이트까지 거리 |
| `info` | dict | 이 구간의 `course.json` 미션 정보 (`zone`, `slot`, `center`, `npc_radius` …) |

**출력 `MissionOutput`**: 기본값은 모두 "아무것도 안 바꿈"

| 필드 | 값 | 효과 |
|---|---|---|
| `override_path` | `None` 또는 `[(x, y), ...]` / `[(x, y, yaw), ...]` (점 2개 이상) | 기본 경로 대신 이 경로를 Stanley/PI가 따라감 |
| `override_cmd` | `None` 또는 `(throttle 0~1, brake 0~1, steer -1~1, gear)` | control이 이 명령을 그대로 보냄. `gear`: `'D'` `'R'` `'N'` `'P'`. steer는 +가 왼쪽, 1 = 35.4° |
| `is_done` | bool | `True`면 그 구간이 끝날 때까지 다시 부르지 않음 |

- `override_cmd`를 그만 보내면 0.2 s(`override_timeout_s`) 뒤 평소 제어로 돌아갑니다.
- 예: 후진 `(0.2, 0.0, -0.5, 'R')`, 정지 `(0.0, 1.0, 0.0, 'D')`

### 팀원 1 → 미션 데이터 약속

`lidar_perception.py`는 장애물·주차공간 점을 **`/hk/perception/obstacles` (`geometry_msgs/PoseArray`, `frame_id: map`)** 로 publish합니다. 각 Pose의 `position.x, y`를 쓰면 미션의 `inp.obstacles`로 자동 전달됩니다.

`lidar_perception`은 기본 실행에서 뜨지 않으므로 다음처럼 켜서 실행하세요.

```bash
ros2 launch hk_mit bringup.launch.py perception:=true
```

### CP 구간 혼자 시험해 보기 (ROS 없이)

```bash
cd mit_ws/src/hk_mit
python3 -c "from hk_mit.cp_manager import CPManager; cp=CPManager('course/course.json','course/route.csv'); cp.reset(8); print(cp.update(-13.16, 130.0, -1.57))"
```

---

## 4. 협업 팁: 기능별 브랜치로 작업하기

`main`에 바로 push하지 말고, 기능별 브랜치를 만들어 작업한 뒤 Pull Request로 합쳐 주세요.

```bash
git checkout main
git pull
git checkout -b feature/planner-tuning   # 예: feature/lidar, fix/control-bug

# 작업 후
git add .
git commit -m "planner: lookahead 거리 조정"
git push -u origin feature/planner-tuning
```

이후 GitHub에서 Pull Request를 열어 팀원 리뷰를 받고 `main`에 merge합니다.

- 같은 파일을 여러 명이 동시에 고치지 않도록 담당 모듈(localization / planner / control / perception)을 나누면 충돌이 줄어듭니다.
- 작업을 시작하기 전에 항상 `git pull`로 `main`을 최신 상태로 맞추세요.
- `build/`, `install/`, `log/`, `data/`는 `.gitignore`로 제외되어 있으니 커밋하지 마세요.
