import json
import socket


class TCPUitility:
    @staticmethod
    def eof_check(message):
        eof_format = "<EOF>"
        message_len = len(message)
        if message_len > 5:
            for i in range(5):
                if eof_format[i] != message[message_len + i - 5]:
                    return False
            return True
        return False

    @staticmethod
    def get_message(client):
        message = ""
        while (True):
            data = client.recv(1024)
            message += (data.decode('UTF-8'))
            if TCPUitility.eof_check(message):
                message = message[:-5]
                break
        dic = json.loads(message)
        return dic

    @staticmethod
    def send_message(client, dic):
        message = json.dumps(dic) + "<EOF>"
        client.send(bytearray(message, 'UTF-8'))


class TCPConnection:
    def __init__(self, port=7979):
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('localhost', port))
        self.server.listen()
        (self.client, address) = self.server.accept()

    def get_client(self):
        return self.client

    def destroy(self):
        self.server.close()
