# OAK-D PoE Camera Network Setup Guide

## Problem
After power cycling the OAK-D PoE camera and IPC, the camera becomes unreachable and DepthAI cannot discover or connect to it.

## Root Cause
The OAK-D PoE camera uses a **static fallback IP address** (`169.254.1.222`) when no DHCP server is available. This requires the host interface to have a link-local IP in the same subnet (`169.254.0.0/16`).

After a power cycle, the interface may lose this configuration, preventing communication with the camera.

## Solution

### 1. Identify the Camera Interface
First, check which network interface the camera is connected to:

```bash
ip addr show
```

In our case: **enp2s0f0** (connected to the camera)

### 2. Verify Camera Presence (Layer 2)
Check if the camera is visible on the network:

```bash
arp -a | grep -i 169.254
```

Expected output:
```
? (169.254.1.222) at 44:a9:2c:30:69:91 [ether] on enp2s0f0
```

If the camera appears in ARP table but is not pingable, proceed to next step.

### 3. Add Link-Local IP Address
Add a link-local IP to the interface connected to the camera:

```bash
sudo ip addr add 169.254.1.1/16 dev enp2s0f0
```

### 4. Test Connectivity
Verify the camera is now reachable:

```bash
# Test with ping
ping -c 2 169.254.1.222

# Test with DepthAI
python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); print(f'Found {len(devices)} camera(s)'); [print(f'  - {d.name}') for d in devices]"
```

Expected output:
```
Found 1 camera(s)
  - 169.254.1.222
```

### 5. Make Configuration Permanent
To ensure the configuration survives reboots, add the IP permanently using NetworkManager:

```bash
# Add the link-local IP to the connection
sudo nmcli connection modify "netplan-enp2s0f0" +ipv4.addresses "169.254.1.1/16"

# Apply the configuration
sudo nmcli connection up "netplan-enp2s0f0"

# Verify it's saved
nmcli connection show "netplan-enp2s0f0" | grep ipv4.addresses
```

Expected output:
```
ipv4.addresses:    192.168.1.100/24, 169.254.1.1/16
```

## Verification Script
Use the provided test script to verify camera connectivity:

```bash
python3 test_camera_connection.py
```

Expected output:
```
✓ Successfully connected to camera!
  Device name: OAK-D-PRO-POE
  MxID: 1944301011A0F14800
```

## Network Configuration Summary

| Component | IP Address | Interface | Notes |
|-----------|------------|-----------|-------|
| OAK-D Camera | 169.254.1.222 | - | Static fallback IP (no DHCP) |
| IPC | 169.254.1.1/16 | enp2s0f0 | Link-local for camera communication |
| IPC | 192.168.1.100/24 | enp2s0f0 | Primary network (if DHCP available) |
| IPC | 10.202.8.43/24 | enp6s0 | Secondary network connection |

## Application Integration
The camera service in `backend/src/services/oakd_camera.py` automatically handles connection:

```python
# Line 17: Fallback IP constant
OAKD_POE_FALLBACK_IP = "169.254.1.222"

# Lines 68-81: Auto-discovery with fallback to static IP
devices = dai.Device.getAllAvailableDevices()
for dev in devices:
    if dev.protocol == dai.XLinkProtocol.X_LINK_TCP_IP:
        device_info = dev
        break

if device_info is None:
    logger.info(f"Falling back to static IP: {OAKD_POE_FALLBACK_IP}")
    device_info = dai.DeviceInfo(OAKD_POE_FALLBACK_IP)
```

No code changes are needed - the application will automatically discover and connect to the camera once the network is configured.

## Troubleshooting

### Camera Not Visible in ARP Table
```bash
# Check if interface is up
ip link show enp2s0f0

# Bring interface up if down
sudo ip link set enp2s0f0 up

# Wait 10-15 seconds for camera to boot
sleep 15
```

### Camera Visible but Not Connecting
```bash
# Check routing
ip route show

# Verify link-local IP is assigned
ip addr show enp2s0f0 | grep "169.254"

# Test layer 2 connectivity
arping -I enp2s0f0 -c 3 169.254.1.222
```

### DepthAI Cannot Find Camera
```bash
# Check DepthAI version
python3 -c "import depthai as dai; print(dai.__version__)"

# List all devices including bootloader state
python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); [print(f'{d.name} - {d.state.name} - {d.protocol.name}') for d in devices]"
```

### After Reboot Camera Not Working
```bash
# Verify NetworkManager configuration is active
nmcli connection show "netplan-enp2s0f0" | grep ipv4.addresses

# If missing, re-add the configuration
sudo nmcli connection modify "netplan-enp2s0f0" +ipv4.addresses "169.254.1.1/16"
sudo nmcli connection up "netplan-enp2s0f0"
```

## Quick Recovery After Power Cycle

If the camera becomes unreachable after a power cycle:

```bash
# One-line fix (temporary)
sudo ip addr add 169.254.1.1/16 dev enp2s0f0

# Verify
ping -c 2 169.254.1.222
```

## References
- [OAK-D PoE Deployment Guide](https://docs.luxonis.com/hardware/platform/deploy/poe-deployment-guide/)
- Camera MAC Address: `44:a9:2c:30:69:91`
- Camera Model: OAK-D-PRO-POE
- MxID: `1944301011A0F14800`
