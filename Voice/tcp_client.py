import socket

HOST = '127.0.0.1'   # atau IP server, misal '192.168.1.10'
PORT = 5005          # harus sama dengan port di C# server

def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(command.encode('utf-8'))
            print(f"Sent: {command}")
    except ConnectionRefusedError:
        print("Connection refused. Is the server running?")
    except Exception as e:
        print(f"Error: {e}")

# Contoh pengiriman perintah
if __name__ == "__main__":
    while True:
        cmd = input("Enter command (left, right, jump, stop): ").strip()
        if cmd.lower() == "exit":
            break
        send_command(cmd)