# src/test_runner.py

import tempfile, subprocess, os

def run_tests_on_code(code_str):
    """
    1) Write code to temp file
    2) Check syntax or run real tests
    3) Return (passed: bool, logs: str)
    """
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            tmp_file = tmp.name
            tmp.write(code_str.encode("utf-8"))

        cmd = f"python -m py_compile {tmp_file}"
        proc = subprocess.run(cmd.split(), capture_output=True, text=True)
        if proc.returncode != 0:
            return False, proc.stderr

        return True, "Syntax OK"
    except Exception as e:
        return False, str(e)
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.remove(tmp_file)
