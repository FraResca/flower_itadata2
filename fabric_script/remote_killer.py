from fabric import Connection, task
import threading
from invoke import Context
from invoke import Responder
import json

# Define each terminal's task
def run_commands(host, commands, sudo_password, is_local=False):

    conn = Connection(host)

    sudopass = Responder(pattern=r'\[sudo\] password*', response=f'{sudo_password}\n')
    
    for cmd in commands:
        print(f"\n[{host}] Running: {cmd}")
        #running the command on the remote host with sudo privileges
        result = conn.run(cmd, hide=False, pty=True, watchers=[sudopass])
        print(f"[{host}] Output:\n{result.stdout}")


killer_commands = ["sudo kill -9 $(pidof python3)"]


# Host and command configurations
with open("sudo_passwds.json") as f:
    sudo_passwds = json.load(f)

terminals = [
    {"host": "rambo", "commands": killer_commands, "sudo_password": sudo_passwds["rambo"]},
    {"host": "giordano", "commands": killer_commands, "sudo_password": sudo_passwds["giordano"]},
    {"host": "girolamo", "commands": killer_commands, "sudo_password": sudo_passwds["girolamo"]}
]

@task
def run_all(c):
    threads = []
    for term in terminals:
        t = threading.Thread(
            target=run_commands,
            args=(term["host"], term["commands"], term["sudo_password"]),
        )
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

def main():
    run_all(Context())

if __name__ == "__main__":
    print("Starting remote killer script...")
    print("This script will kill all python3 processes on the specified hosts.")
    main()
