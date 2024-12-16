with open('requirements.txt', 'r') as file:
    lines = file.readlines()

with open('requirements.txt', 'w') as file:
    for line in lines:
        package, version = line.strip().split()
        file.write(f"{package}=={version}\n")