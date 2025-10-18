#!/bin/bash
# Setup S3 uploader service - Run this to create the 24/7 scheduled service

echo "Creating S3 uploader service..."

# Create service file
sudo tee /etc/systemd/system/s3-uploader.service > /dev/null <<'EOF'
[Unit]
Description=S3 Daily Upload Service
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=denaliai
Group=denaliai
WorkingDirectory=/home/denaliai/RELO-DEMO-APP-latest/S3-bucket
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 /home/denaliai/RELO-DEMO-APP-latest/S3-bucket/s3_uploader_scheduled.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=s3-uploader

[Install]
WantedBy=multi-user.target
EOF

# Create timer file
sudo tee /etc/systemd/system/s3-uploader.timer > /dev/null <<'EOF'
[Unit]
Description=S3 Daily Upload Timer
Requires=s3-uploader.service

[Timer]
OnCalendar=*-*-* 01:00:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

# Reload systemd
sudo systemctl daemon-reload

echo ""
echo "✅ Service created successfully!"
echo ""
echo "⚠️  Service is NOT started yet (manual control only)"
echo ""
echo "To start the service:"
echo "  sudo systemctl start s3-uploader.timer"
echo ""
echo "To enable auto-start on boot (optional):"
echo "  sudo systemctl enable s3-uploader.timer"
echo ""
echo "See README.md for all control commands."
