from fabric import Connection, task
import threading
from invoke import Context


# Define each terminal's task
def run_commands(host, commands, is_local=False):

    conn = Connection(host)
    
    for cmd in commands:
        print(f"\n[{host}] Running: {cmd}")
        result = conn.run(cmd, hide=False)
        print(f"[{host}] Output:\n{result.stdout}")


killer_commands = ["kill -9 $(pidof python3)"]


# Host and command configurations
terminals = [
    {"host": "rambo", "commands": killer_commands},
    {"host": "giordano", "commands": killer_commands},
    {"host": "girolamo", "commands": killer_commands}
]

@task
def run_all(c):
    threads = []
    for term in terminals:
        t = threading.Thread(
            target=run_commands,
            args=(term["host"], term["commands"]),
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
