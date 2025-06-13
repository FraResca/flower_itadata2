from fabric import Connection, task
import threading
from invoke import Context

model = "360"
alfa = "01"

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
    "cd repos/flower_itadata2 && git pull && source venv_flower/bin/activate && ./startclient.sh D {model} {alfa}".format(model=model, alfa=alfa),
]

commands_clientB = [
    "cd Scaricati/flower_itadata2 && git pull && source venv_flower/bin/activate && ./startclient.sh E {model} {alfa}".format(model=model, alfa=alfa),
]

commands_clientC = [
    "cd flower_itadata2 && git pull && source venv_flower/bin/activate && ./startclient.sh F {model} {alfa}".format(model=model, alfa=alfa),
]

commands_server = [
    "cd repos/flower_itadata2 && git pull && source venv_flower/bin/activate && ./startserver.sh {model} {alfa}".format(model=model, alfa=alfa),
]

# Host and command configurations
terminals = [
    {"host": "rambo", "commands": commands_server},
    {"host": "rambo", "commands": commands_clientA},
    {"host": "giordano", "commands": commands_clientB},
    {"host": "girolamo", "commands": commands_clientC},
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

def main():
    run_all(Context())

if __name__ == "__main__":
    main()
