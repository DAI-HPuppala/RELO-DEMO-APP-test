#!/usr/bin/env python3
"""Test script to discover CV60 camera using eBUS SDK."""
import os
import sys
import site

# Setup eBUS SDK paths (same as cv60_camera.py)
for p in site.getsitepackages():
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

for path in [
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3.10/dist-packages",
    "/usr/local/lib/python3.10/site-packages",
    "/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib"
]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib'

try:
    import eBUS as eb
    print("✅ eBUS SDK imported successfully")
except ImportError as e:
    print(f"❌ ERROR: Failed to import eBUS SDK: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("DISCOVERING GIGE VISION CAMERAS")
print("="*80)

# Create PvSystem for device discovery
system = eb.PvSystem()

print("\n🔍 Searching for GigE Vision devices...")
system.Find()

# Get interface count
interface_count = system.GetInterfaceCount()
print(f"\n📡 Found {interface_count} network interfaces")

# Enumerate all network interfaces
for i in range(interface_count):
    iface = system.GetInterface(i)

    # Get interface info
    iface_name = iface.GetName()
    iface_display_id = iface.GetDisplayID()

    print(f"\n{'─'*80}")
    print(f"Interface {i}: {iface_display_id}")
    print(f"  Name: {iface_name}")

    # Try to get IP address
    try:
        ip = str(iface.GetIPAddress(0))
        print(f"  IP: {ip}")
    except Exception as e:
        print(f"  IP: N/A ({e})")

    # Get device count on this interface
    device_count = iface.GetDeviceCount()
    print(f"  Devices on this interface: {device_count}")

    # Enumerate devices on this interface
    for d in range(device_count):
        device_info = iface.GetDeviceInfo(d)

        print(f"\n  {'┌'*40}")
        print(f"  Device {d}:")
        print(f"  {'├'*40}")

        # Get device details
        try:
            display_id = device_info.GetDisplayID()
            print(f"    Display ID: {display_id}")
        except:
            pass

        try:
            vendor = device_info.GetVendorName()
            print(f"    Vendor: {vendor}")
        except:
            pass

        try:
            model = device_info.GetModelName()
            print(f"    Model: {model}")
        except:
            pass

        try:
            serial = device_info.GetSerialNumber()
            print(f"    Serial: {serial}")
        except:
            pass

        try:
            user_defined_name = device_info.GetUserDefinedName()
            print(f"    User Name: {user_defined_name}")
        except:
            pass

        try:
            mac = device_info.GetMACAddress()
            print(f"    MAC: {mac}")
        except:
            pass

        try:
            device_ip = device_info.GetIPAddress()
            print(f"    IP Address: {device_ip}")
        except:
            pass

        try:
            connection_id = device_info.GetConnectionID()
            print(f"    ⭐ CONNECTION ID: {connection_id}")
        except:
            pass

        print(f"  {'└'*40}")

print("\n" + "="*80)
print("DISCOVERY COMPLETE")
print("="*80)

# Now test connection to first found device
print("\n🔌 Testing connection to first discovered device...")

# Find the first device
device_to_connect = None
connection_string = None

for i in range(interface_count):
    iface = system.GetInterface(i)
    device_count = iface.GetDeviceCount()

    if device_count > 0:
        device_info = iface.GetDeviceInfo(0)
        try:
            connection_string = device_info.GetConnectionID()
            print(f"   Found device with connection ID: {connection_string}")
            break
        except:
            pass

if connection_string:
    print(f"\n🔗 Attempting to connect to: {connection_string}")

    try:
        result, device = eb.PvDevice.CreateAndConnect(connection_string)

        if device:
            print("✅ CONNECTION SUCCESSFUL!")
            print(f"   Device connected: {device}")

            # Try to get some device parameters
            try:
                params = device.GetParameters()
                print("\n📊 Device Parameters:")
                print(f"   Payload Size: {device.GetPayloadSize()} bytes")
            except Exception as e:
                print(f"   Could not read parameters: {e}")

            # Disconnect
            device.Disconnect()
            print("   Device disconnected")
        else:
            print(f"❌ CONNECTION FAILED: CreateAndConnect returned None")
            print(f"   Result: {result}")

    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")
else:
    print("❌ No devices found to connect to")

print("\n" + "="*80)
