import os


def load_env_file(file_path='.env'):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value and (value[0] == value[-1] == '"' or value[0] == value[-1] == "'"):
                    value = value[1:-1]

                os.environ[key] = value
