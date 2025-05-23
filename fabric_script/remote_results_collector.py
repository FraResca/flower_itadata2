from fabric import Connection, task
import threading
from invoke import Context
import os

'''
This script collects files from remote servers using Fabric.
It connects to each server, runs a command to navigate to a specific directory, collects files with specified extensions, and saves them locally in a folder with same same name of the host.
'''

extensions_to_collect = [".txt", ".lore", ".jsonl"]

# Define each terminal's task
def get_files(host, command):
    print(f"{host}, {type(host)}")
    print(f"{command}, {type(command)}")
    conn = Connection(host)
    # Create a directory to store collected files in the local machine
    local_dir = f"./collected_files/{host}/"
    os.makedirs(local_dir, exist_ok=True)

    #remove all files in the local directory
    for file in os.listdir(local_dir):
        os.remove(os.path.join(local_dir, file))

    print(f"\n[{host}] Running: {command}", type(command))
    result = conn.run(command, hide=False)
 
    # Collect files with specified extensions
    for ext in extensions_to_collect:
        #files = conn.run(f"find . -maxdepth 1 -name '*{ext}'")
        files = conn.run("pwd")
        if files.stdout:
            print(f"[{host}] Found files with extension {ext}: {files.stdout}")
            # Copying files to local machine
            #conn.get(files.stdout, local=f"./collected_files/{host}/")
        else:
            print(f"[{host}] No files found with extension {ext}")

    print(f"[{host}] Output:\n{result.stdout}")


commands_clientB =  "cd Scaricati/flower_itadata2"
commands_clientC =  "cd flower_itadata2"
commands_server =  "cd flower_itadata2"


# Host and command configurations
terminals = [
    {"host": "giordano", "command": commands_clientB},
    {"host": "girolamo", "command": commands_clientC},
    {"host": "rambo", "command": commands_server},
]

@task
def run_all(c):
    threads = []
    for term in terminals:
      
        t = threading.Thread(
            target=get_files,
            args=(term["host"], term["command"]),
        )
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    run_all(Context())