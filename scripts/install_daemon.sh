#!/bin/bash
# 安装 embers daemon 的 launchd 定时任务
#   每天 0:00 / 12:00 触发;关机错过的,开机后 launchd 自动补跑一次
#   daemon 内部 state 判断"距上次 <11h 跳过",防开机频繁重复
#
# 用法:
#   bash scripts/install_daemon.sh            # 只装 plist,不激活(默认,安全)
#   bash scripts/install_daemon.sh --activate # 装并立即 launchctl load 激活
#   bash scripts/install_daemon.sh --uninstall

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJECT_ROOT/.venv/bin/python"
LABEL="com.embers.daemon"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "$1" = "--uninstall" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "✓ 已卸载 $LABEL"
    exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PY</string>
        <string>-m</string>
        <string>pipeline.daemon</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>CLIP_DEVICE</key>
        <string>mps</string>
    </dict>

    <!-- 每天 0:00 和 12:00;关机错过的 launchd 会在开机后补跑一次 -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>0</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>12</integer><key>Minute</key><integer>0</integer></dict>
    </array>

    <!-- 加载/开机即跑一次(daemon 内部 state <11h 判断防重复) -->
    <key>RunAtLoad</key>
    <true/>

    <!-- 一次性任务:跑完退出,不常驻 -->
    <key>KeepAlive</key>
    <false/>

    <key>StandardOutPath</key>
    <string>$HOME/.embers_daemon.launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/.embers_daemon.launchd.log</string>
</dict>
</plist>
EOF

echo "✓ 已写入 plist: $PLIST"
echo "  Python: $PY"
echo "  项目:   $PROJECT_ROOT"

if [ "$1" = "--activate" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "✓ 已激活 (launchctl load)。RunAtLoad 会马上触发一轮"
    echo "  看日志: tail -f ~/.embers_daemon.log"
else
    echo
    echo "未激活(安全默认)。确认无误后激活:"
    echo "  launchctl load \"$PLIST\""
    echo "卸载: bash scripts/install_daemon.sh --uninstall"
fi
