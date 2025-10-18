#!/bin/bash
# update_schedule.sh - Apply config.py changes to systemd timer
# This script updates the timer schedule from config.py without triggering an upload

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.py"
TIMER_FILE="/etc/systemd/system/s3-uploader.timer"

echo "=========================================="
echo "S3 Uploader Schedule Update Script"
echo "=========================================="
echo ""

# Check if config.py exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: config.py not found at $CONFIG_FILE"
    exit 1
fi

# Extract JOB_START_TIME from config.py
echo "📖 Reading schedule from config.py..."
JOB_START_TIME=$(python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
import config
print(config.JOB_START_TIME)
")

if [ -z "$JOB_START_TIME" ]; then
    echo "❌ Error: Could not read JOB_START_TIME from config.py"
    exit 1
fi

echo "   JOB_START_TIME: $JOB_START_TIME"
echo ""

# Validate time format (HH:MM)
if ! [[ "$JOB_START_TIME" =~ ^[0-2][0-9]:[0-5][0-9]$ ]]; then
    echo "❌ Error: Invalid time format '$JOB_START_TIME'. Expected HH:MM (24-hour format)"
    exit 1
fi

# Convert HH:MM to systemd OnCalendar format (*-*-* HH:MM:00)
ONCALENDAR_TIME="*-*-* ${JOB_START_TIME}:00"

echo "🔧 Updating systemd timer..."
echo "   New schedule: Daily at $JOB_START_TIME"
echo ""

# Check if timer file exists
if [ ! -f "$TIMER_FILE" ]; then
    echo "❌ Error: Timer file not found at $TIMER_FILE"
    echo "   Run ./setup_service.sh first to create the service"
    exit 1
fi

# Update the timer file (requires sudo)
sudo tee "$TIMER_FILE" > /dev/null <<EOF
[Unit]
Description=S3 Daily Upload Timer
Requires=s3-uploader.service

[Timer]
OnCalendar=$ONCALENDAR_TIME
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

echo "✅ Timer file updated successfully"
echo ""

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload
echo "✅ Daemon reloaded"
echo ""

# Check if timer is currently active
TIMER_ACTIVE=$(systemctl is-active s3-uploader.timer 2>/dev/null || echo "inactive")

if [ "$TIMER_ACTIVE" = "active" ]; then
    echo "🔄 Restarting timer (does NOT trigger upload)..."
    sudo systemctl restart s3-uploader.timer
    echo "✅ Timer restarted"
else
    echo "ℹ️  Timer is not running (status: $TIMER_ACTIVE)"
    echo "   To start it: sudo systemctl start s3-uploader.timer"
fi

echo ""
echo "=========================================="
echo "✅ Configuration Applied Successfully!"
echo "=========================================="
echo ""

# Show next scheduled run
echo "📅 Next scheduled run:"
systemctl list-timers s3-uploader.timer --no-pager 2>/dev/null | grep -A 1 "NEXT" || echo "   (Timer not active)"
echo ""

# Show current timer status
echo "📊 Timer status:"
systemctl status s3-uploader.timer --no-pager -l 2>/dev/null | head -n 5 || echo "   (Timer not found)"
echo ""

echo "ℹ️  Notes:"
echo "   - Config changes in config.py are automatically picked up on next run"
echo "   - This script only needs to be run if you changed JOB_START_TIME"
echo "   - Restarting the timer does NOT trigger an immediate upload"
echo "   - Upload will only trigger at the scheduled time: $JOB_START_TIME"
echo ""
