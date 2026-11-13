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

## 3. 협업 팁: 기능별 브랜치로 작업하기

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
