message = "Hello world!"
import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 4848        # The port used by the server

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
client.sendall(b'Hello, world')