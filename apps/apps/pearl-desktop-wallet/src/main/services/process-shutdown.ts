import {spawn} from 'child_process';
import type {ChildProcess} from 'child_process';

interface ShutdownOptions {
  // Asks the process to exit by itself. It has to settle within stopTimeoutMs.
  requestStop: () => Promise<unknown>;
  stopTimeoutMs: number;
  // How long the process gets after the termination request, and after the kill.
  signalTimeoutMs: number;
  // For callers that can't wait for a process that is slow to exit.
  skipTerminate?: boolean;
}

function hasExited(proc: ChildProcess): boolean {
  return proc.exitCode !== null || proc.signalCode !== null;
}

function waitForExit(proc: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (hasExited(proc)) {
    return Promise.resolve(true);
  }

  return new Promise<boolean>(resolve => {
    const onExit = () => {
      clearTimeout(timer);
      resolve(true);
    };
    const timer = setTimeout(
      () => {
        proc.off('exit', onExit);
        resolve(false);
      },
      Math.max(timeoutMs, 0)
    );
    proc.once('exit', onExit);
  });
}

// taskkill without /F asks the process to close its windows. Console programs have none, for which it fails at once,
// so the exit code tells whether waiting for the process is worthwhile.
function closeWindowsProcess(pid: number): Promise<boolean> {
  return new Promise<boolean>(resolve => {
    const taskkill = spawn('taskkill', ['/PID', String(pid), '/T'], {stdio: 'ignore', windowsHide: true});
    taskkill.once('error', () => resolve(false));
    taskkill.once('exit', code => resolve(code === 0));
  });
}

// On Windows every signal sent through the child handle is a hard kill, so the termination request goes through
// taskkill instead. Resolves true once the process has exited.
async function shutdownProcess(proc: ChildProcess, options: ShutdownOptions): Promise<boolean> {
  if (proc.pid === undefined || hasExited(proc)) {
    return true;
  }

  const stopDeadline = Date.now() + options.stopTimeoutMs;
  let stopDelivered = true;
  try {
    await options.requestStop();
  } catch (error) {
    stopDelivered = false;
    console.warn(`Could not ask process ${proc.pid} to stop: ${error instanceof Error ? error.message : error}`);
  }
  if (await waitForExit(proc, stopDelivered ? stopDeadline - Date.now() : 0)) {
    return true;
  }

  if (!options.skipTerminate) {
    const terminating = process.platform === 'win32' ? await closeWindowsProcess(proc.pid) : proc.kill('SIGTERM');
    if (await waitForExit(proc, terminating ? options.signalTimeoutMs : 0)) {
      return true;
    }
  }

  console.warn(`Process ${proc.pid} is still running, killing it`);
  proc.kill('SIGKILL');
  return waitForExit(proc, options.signalTimeoutMs);
}

export {shutdownProcess};
