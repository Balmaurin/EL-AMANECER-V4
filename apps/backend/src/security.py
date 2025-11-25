class SecurityError(Exception):
    pass

def validate_command_args(args):
    pass

def validate_timeout(timeout):
    pass

def sanitize_filename(filename):
    return filename.replace("..", "").replace("/", "").replace("\\", "")
