import socket
import time

HOST = "rl-research-kinovagen3-wato_research_kinova-1"  # Standard loopback interface address (localhost)
PORT = 65432 # Port to listen on (non-privileged ports are > 1023)

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         # print("Connected by ")
#         while True:
#             data = conn.recv(1024).decode("utf-8")
#             positions = list(map(float, data.split(",")))
#             print(positions)
#             if not data:
#                 break
#             # conn.sendall(data)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()


def get_command():
    global conn, addr

    positions = [0., 0., 0.]
    restart = False
    try:
        while True:
            data = conn.recv(1024)
            d = data.decode("utf-8")
            if d == "reset":
                print("restart")
                restart = True
            else:
                positions = list(map(float, d.split(",")))
                print(positions)
            conn.sendall(b"done\n")
            if d:
                break
    except:
        conn, addr = s.accept()
    
    return positions, restart
            
def send_observation(observations):
    obs = ",".join(map(str, observations)) + "\n"
    conn.sendall(obs.encode('utf-8'))


# class communicator:
#     def __init__(self, host = HOST, port = PORT):

        