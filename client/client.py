import socket
import time

HOST = "rl-research-kinovagen3-wato_research_kinova-1"
PORT = 65432

class Robot:
    def __init__(self, host = HOST, port = PORT):
        self.host = host
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self.host, self.port))

    def send_receive(self, ee_delta: list) -> list:
        self._send_ee_delta(ee_delta)
        return self._receive_arm_observation()
    
    def _send_ee_delta(self, ee_delta: list):
        ee_delta = list(map(str, ee_delta))
        string = ",".join(ee_delta)
        self._socket.sendall(string.encode("utf-8"))

    def _receive_arm_observation(self):
        while True:
            data = self._socket.recv(1024)
            observations = data.decode('utf-8')
            observations = observations.strip().split("\n")[-1]
            if (observations == 'done'):
                continue
            else:
                obs = list(map(float, observations.split(",")))
                return obs

    def disconnect(self):
        self._socket.close()

    def reset(self):
        string = "reset"
        self._socket.sendall(string.encode("utf-8"))
        success = self._socket.recv(1024).decode('utf-8')
        success = success
        print(success)


waypoints = [[0.0, 0.0, -0.1], [0.0, 0.0, 0.1],[0.0, 0.0, -0.01], [0.0, 0.0, 0.01],[0.0, 0.0, -0.1], [0.0, 0.0, 0.2]]
if __name__ == "__main__":
    time.sleep(10)
    r = Robot()
    for point in waypoints:
        r.send_receive(point)
    r.disconnect()