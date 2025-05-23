from fabric import Connection, task
import threading

model = "135"
alfa = "05"

# Define each terminal's task
def run_commands(host, commands, is_local=False):
    if is_local:
        conn = Connection('localhost')
    else:
        conn = Connection(host)
    
    for cmd in commands:
        print(f"\n[{host}] Running: {cmd}")
        result = conn.run(cmd, hide=False)
        print(f"[{host}] Output:\n{result.stdout}")



commands_clientA = [
    "cd repos/flower_itadata2",
    "conda activate flower",
    f"./startclient.sh A {model}",
]

commands_clientB = [
    "cd Scaricati/flower_itadata2",
    "source venv_flower/bin/activate",
    f"./startclient.sh B {model}",
]

commands_clientC = [
    "cd Scaricati/flower_itadata2",
    "source venv_flower/bin/activate",
    f"./startclient.sh C {model}",
]

commands_server = [
    "cd repos/flower_itadata2",
    "conda activate flower",
    f"./startserver.sh {model} {alfa}",
]

# Host and command configurations
terminals = [
    {"host": "rambo", "commands": commands_clientA},      # Remote 1
    {"host": "giordano", "commands": commands_clientB},     # Remote 2
    {"host": "girolamo", "commands": commands_clientC},# Remote 3
    {"host": "rambo",     "commands": commands_server, "local": True},  # Local
]

@task
def run_all(c):
    threads = []
    for term in terminals:
        t = threading.Thread(
            target=run_commands,
            args=(term["host"], term["commands"]),
            kwargs={"is_local": term.get("local", False)}
        )
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()