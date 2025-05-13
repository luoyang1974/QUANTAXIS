import shlex
import subprocess
import sys

from QUANTAXIS.QAUtil.QALogs import QA_util_log_info


def run_backtest(shell_cmd):
    shell_cmd = f'python "{shell_cmd}"'
    cmd = shlex.split(shell_cmd)
    p = subprocess.Popen(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    while p.poll() is None:
        if p.stdout is None:
            continue
        line = p.stdout.readline()
        if not line:
            continue
        line = line.strip().decode('utf-8')
        if line:
            QA_util_log_info(line)
            #print('QUANTAXIS: [{}]'.format(line))
    if p.returncode == 0:
        QA_util_log_info('backtest run  success')

    else:
        QA_util_log_info('Subprogram failed')
    return p.returncode


def run():
    shell_cmd = sys.argv[1]
    print(shell_cmd)
    return run_backtest(shell_cmd)


if __name__ == "__main__":
    print(run())
