from fabric import Connection

c = Connection("rambo")
result = c.run("python")
result = c.run("echo ciao")

