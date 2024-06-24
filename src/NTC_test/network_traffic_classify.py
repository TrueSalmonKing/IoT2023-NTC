from scapy.all import *
import threading
import time
import pandas as pd
import pickle

datasets_dir = 'training_data/'
flows = {}

# Dictionary to keep track of the last sequence numbers for TCP flows
last_seq_tracker = {}

# Lock to ensure thread-safe operations
lock = threading.Lock()

# Function to calculate missed bytes in a TCP flow
def calculate_missed_bytes(packet, flow_key):
    if flow_key not in last_seq_tracker:
        last_seq_tracker[flow_key] = packet[TCP].seq
        return 0
    
    last_seq = last_seq_tracker[flow_key]
    current_seq = packet[TCP].seq
    
    if current_seq > last_seq:
        missed_bytes = current_seq - last_seq - len(packet[TCP].payload)
    else:
        missed_bytes = 0
    
    last_seq_tracker[flow_key] = current_seq
    return max(missed_bytes, 0)

# Function to classify a flow based on the captured features
def classify(clf, enc, flow):
    duration = time.time() - flow['start_time']
    
    # Convert the flow to a DataFrame
    flow_df = pd.DataFrame([{
        'proto': flow['proto'].lower(),
        'src_bytes': str(flow['src_bytes']),
        'dst_bytes': flow['dst_bytes'],
        'conn_state': flow['conn_state'],
        'missed_bytes': flow['missed_bytes'],
        'src_ip_bytes': flow['src_bytes'],
        'dst_ip_bytes': flow['dst_bytes']
    }])
    
    # Encode the flow DataFrame
    #(flow_df.dtypes)
    X = enc.transform(flow_df)
    
    # Predict the classification
    prediction = clf.predict(X)
    
    # Print the classification result
    print(f"Flow from {flow['src_ip']} to {flow['dst_ip']} classified as {prediction[0]}.")
    #print(f"Duration: {duration:.2f} seconds")
    #print(f"Source Bytes: {flow['src_bytes']}")
    #print(f"Destination Bytes: {flow['dst_bytes']}")
    #print(f"Connection State: {flow['conn_state']}")
    #print(f"Missed Bytes: {flow['missed_bytes']}")
    #print("---------------------------------------------")

# Function to process each captured packet
def process_packet(packet):
    if IP in packet:
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        
        # Determine protocol
        if TCP in packet:
            proto = 'TCP'
            conn_state = packet.sprintf("%TCP.flags%")
            src_bytes = len(packet[TCP].payload)
            dst_bytes = 0  # Will be calculated from responses, if any
            flow_key = (ip_src, ip_dst, 'TCP')
            missed_bytes = calculate_missed_bytes(packet, flow_key)
        elif UDP in packet:
            proto = 'UDP'
            conn_state = 'N/A'  # UDP doesn't have connection state
            src_bytes = len(packet[UDP].payload)
            dst_bytes = 0  # Will be calculated from responses, if any
            missed_bytes = 0  # UDP doesn't have missed bytes
            flow_key = (ip_src, ip_dst, 'UDP')
        else:
            return  # Ignore other protocols

        with lock:
            # Check if flow already exists or create a new flow
            if flow_key in flows:
                flow = flows[flow_key]
                flow['last_seen'] = time.time()
                flow['src_bytes'] += src_bytes
                flow['dst_bytes'] += dst_bytes
                flow['missed_bytes'] += missed_bytes
                flow['conn_state'] = conn_state
            else:
                flows[flow_key] = {
                    'src_ip': ip_src,
                    'dst_ip': ip_dst,
                    'proto': proto,
                    'start_time': time.time(),
                    'last_seen': time.time(),
                    'src_bytes': src_bytes,
                    'dst_bytes': dst_bytes,
                    'missed_bytes': missed_bytes,
                    'conn_state': conn_state
                }

# Function to periodically classify flows and clean up old ones
def periodic_classify():
    with open(os.path.join(datasets_dir,'clf.pkl'), 'rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(datasets_dir,'encoder.pkl'), 'rb') as f:
        enc = pickle.load(f)

    while True:
        time.sleep(1)
        current_time = time.time()
        with lock:
            expired_flows = [key for key, flow in flows.items() if current_time - flow['last_seen'] >= 1]
            for flow_key in expired_flows:
                classify(clf, enc, flows[flow_key])
                #print(len(flows), flows[flow_key]['last_seen'], flows[flow_key]['src_bytes'], flows[flow_key]['dst_bytes'])
                del flows[flow_key]

# Packet sniffing callback function
def packet_callback(packet):
    process_packet(packet)

# Start a background thread to classify flows every 5 seconds
classifier_thread = threading.Thread(target=periodic_classify, daemon=True)
classifier_thread.start()

# Start sniffing packets
sniff(prn=packet_callback, store=0)
