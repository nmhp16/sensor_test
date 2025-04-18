import asyncio
from bleak import discover

async def scan_bluetooth():
    devices = await discover()
    for device in devices:
        print(f"Device: {device.name}, Address: {device.address}")

if __name__ == "__main__":
    print("Scanning for Bluetooth devices...")
    asyncio.run(scan_bluetooth())
