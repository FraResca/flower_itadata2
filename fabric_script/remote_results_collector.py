from fabric import Connection, task
import threading
from invoke import Context
import os

'''
This script collects files from remote servers using Fabric.
It connects to each server, runs a command to navigate to a specific directory, collects files with specified extensions, and saves them locally in a folder with same same name of the host.
'''


# Define each terminal's task
def get_files(host, path, extensions = [".txt", ".lore", ".jsonl"]):
    print(f"{host}, {type(host)}")
    print(f"{path}, {type(path)}")
    conn = Connection(host)
    # Create a directory to store collected files in the local machine
    local_dir = f"./collected_files/{host}/"
    os.makedirs(local_dir, exist_ok=True)

    #remove all files in the local directory
    for file in os.listdir(local_dir):
        os.remove(os.path.join(local_dir, file))
 
    # Collect files with specified extensions
    for ext in extensions:
        command = f"find {path} -maxdepth 1 -name '*{ext}'"
        print(f"\n[{host}] Running: {command}")
        files = conn.run(command, hide=True)
        if files.stdout:
            print(f"[{host}] Found files with extension {ext}: {files.stdout}")
            # Copying files to local machine
            #conn.get(files.stdout, local=f"./collected_files/{host}/")
            for file in files.stdout.splitlines():
                file = file.strip()
                if file:
                    local_path = os.path.join(local_dir, os.path.basename(file))
                    remote_path = os.path.join(path, file)
                    print(f"[{host}] Copying {remote_path} to {local_path}")
                    conn.get(remote_path, local=local_path)
                    print(f"[{host}] Copied {remote_path} to {local_path}")
        else:
            print(f"[{host}] No files found with extension {ext}")

        print(f"[{host}] Output:\n{files.stdout}")

#find_command =  " find . -maxdepth 1 -name '*EXTENSION'"


# Host and command configurations
terminals = [
    {"host": "giordano", "path": "Scaricati/flower_itadata2"},
    {"host": "girolamo", "path": "flower_itadata2"},
    {"host": "rambo", "path": "repos/flower_itadata2"},
]

@task
def run_all(c):
    threads = []
    for term in terminals:
      
        t = threading.Thread(
            target=get_files,
            args=(term["host"], term["path"]),
        )
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    print("Starting remote results collector script")
    print("The files will be collected in the folder ./collected_files/")
    run_all(Context())