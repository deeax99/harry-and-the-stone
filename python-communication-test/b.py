import socket
HOST = '127.0.0.1'  
PORT = 4848  

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()
conn, addr = server.accept()
while (True):
    data = conn.recv(1024)
    if not data:
        break
    print((data.decode('UTF-8')))

