import scapy.all as scapy

def scan_network(ip_address):
    # Get a list of available networks in your vicinity
    networks = scapy.arping(ip_address, verbose=False)
    for network in networks[0]:
        print(network)

if __name__ == "__main__":
    print("Scanning for available networks...")
    scan_network(ip_address="")