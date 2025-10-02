#!/bin/bash

# OAK-D PoE Camera Network Setup
# Automatically detects and configures the network interface for OAK-D camera

OAKD_IP="169.254.1.222"
OAKD_LINK_LOCAL_IP="169.254.1.1/16"
OAKD_MAC="44:a9:2c:30:69:91"
INTERFACE_CACHE="/tmp/oakd_interface_cache"

# Check if CAMERA_TYPE is set to OAKD
CAMERA_TYPE=$(grep "^CAMERA_TYPE=" backend/.env 2>/dev/null | cut -d'=' -f2 | tr -d ' ')
if [ "$CAMERA_TYPE" != "OAKD" ]; then
    # Not using OAK-D, skip setup
    return 0 2>/dev/null || exit 0
fi

# Function to check if camera is pingable
check_oakd_ping() {
    ping -c 1 -W 1 "$OAKD_IP" > /dev/null 2>&1
    return $?
}

# Function to find OAK-D camera interface using ARP
find_oakd_interface() {
    # Try to trigger ARP entries by pinging on each interface
    for iface in $(ip link show | grep -E "^[0-9]+: enp" | cut -d: -f2 | tr -d ' '); do
        # Skip interfaces that are down
        if ! ip link show "$iface" | grep -q "state UP"; then
            continue
        fi

        # Try to ping from this interface
        ping -c 1 -W 1 -I "$iface" "$OAKD_IP" > /dev/null 2>&1

        # Check ARP table for the MAC address on this interface
        if arp -i "$iface" -a 2>/dev/null | grep -iq "$OAKD_MAC"; then
            echo "$iface"
            return 0
        fi
    done
    return 1
}

# Function to configure link-local IP on interface
configure_interface() {
    local iface=$1

    # Check if IP is already configured and active
    if ip addr show "$iface" 2>/dev/null | grep -q "inet.*169.254.1.1"; then
        return 0
    fi

    # Configure via NetworkManager if available
    nmcli connection show "$iface" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Configuring $iface with $OAKD_LINK_LOCAL_IP for OAK-D camera (via NetworkManager)..."
        # Check if address is already in configuration
        if ! nmcli connection show "$iface" | grep -q "169.254.1.1/16"; then
            echo denaliai | sudo -S nmcli connection modify "$iface" +ipv4.addresses "$OAKD_LINK_LOCAL_IP" 2>/dev/null || true
        fi
        # Activate the connection (suppress password prompt output)
        echo denaliai | sudo -S nmcli connection up "$iface" 2>&1 | grep -v "password for" > /dev/null || true
    else
        # Fallback to manual IP configuration
        echo "Configuring $iface with $OAKD_LINK_LOCAL_IP for OAK-D camera (manual)..."
        echo denaliai | sudo -S ip addr add "$OAKD_LINK_LOCAL_IP" dev "$iface" 2>/dev/null || true
    fi

    # Wait for network to stabilize
    sleep 2
}

# Main setup logic
echo -e "${YELLOW}Checking OAK-D camera network setup...${NC}"

# First, check if camera is already reachable
if check_oakd_ping; then
    echo -e "${GREEN}✓ OAK-D camera is reachable at $OAKD_IP${NC}"
    return 0 2>/dev/null || exit 0
fi

# Camera not reachable - try to find and configure
echo -e "${YELLOW}OAK-D camera not reachable, searching for interface...${NC}"

# Check if we have a cached interface
if [ -f "$INTERFACE_CACHE" ]; then
    CACHED_IFACE=$(cat "$INTERFACE_CACHE")
    echo "Trying cached interface: $CACHED_IFACE"

    # Configure the cached interface
    configure_interface "$CACHED_IFACE"

    # Test if it works
    if check_oakd_ping; then
        echo -e "${GREEN}✓ OAK-D camera configured on $CACHED_IFACE${NC}"
        return 0 2>/dev/null || exit 0
    else
        echo "Cached interface didn't work, scanning all interfaces..."
        rm -f "$INTERFACE_CACHE"
    fi
fi

# Find the interface
OAKD_IFACE=$(find_oakd_interface)
if [ -z "$OAKD_IFACE" ]; then
    echo -e "${YELLOW}⚠ Could not detect OAK-D camera interface${NC}"
    echo -e "${YELLOW}  Camera may not be connected or powered on${NC}"
    echo -e "${YELLOW}  The retry logic in the code will handle this${NC}"
    return 0 2>/dev/null || exit 0
fi

echo "Found OAK-D camera on interface: $OAKD_IFACE"

# Configure the interface
configure_interface "$OAKD_IFACE"

# Verify it works
if check_oakd_ping; then
    echo -e "${GREEN}✓ OAK-D camera successfully configured on $OAKD_IFACE${NC}"
    # Cache the interface for next time
    echo "$OAKD_IFACE" > "$INTERFACE_CACHE"
else
    echo -e "${YELLOW}⚠ Camera detected but not pingable yet${NC}"
    echo -e "${YELLOW}  The retry logic in the code will handle this${NC}"
fi

return 0 2>/dev/null || exit 0
