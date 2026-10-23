import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {test} from 'node:test';

import {loadSource} from './load-source.mjs';

// Every spawned command is passed to runCommand, which returns its exit code.
let runCommand = () => 0;
const childProcess = {
  spawn(command, args) {
    const child = new EventEmitter();
    const code = runCommand(command, args);
    setImmediate(() => child.emit('exit', code));
    return child;
  },
};

const {shutdownProcess} = loadSource({child_process: childProcess})('src/main/services/process-shutdown.ts');

class FakeProcess extends EventEmitter {
  constructor(exitsOn = []) {
    super();
    this.pid = 4242;
    this.exitCode = null;
    this.signalCode = null;
    this.exitsOn = exitsOn;
    this.signals = [];
  }

  kill(signal) {
    this.signals.push(signal);
    if (this.exitsOn.includes(signal)) this.exit(null, signal);
    return true;
  }

  exit(code, signal = null) {
    this.exitCode = code;
    this.signalCode = signal;
    this.emit('exit', code, signal);
  }
}

async function shutdown(proc, {platform = 'linux', ...options} = {}) {
  const realPlatform = Object.getOwnPropertyDescriptor(process, 'platform');
  Object.defineProperty(process, 'platform', {...realPlatform, value: platform});
  try {
    return await shutdownProcess(proc, {
      requestStop: async () => {},
      stopTimeoutMs: 50,
      signalTimeoutMs: 50,
      ...options,
    });
  } finally {
    Object.defineProperty(process, 'platform', realPlatform);
  }
}

test('a process that exits when asked to stop is not signalled', async () => {
  const proc = new FakeProcess();
  const requestStop = async () => {
    setTimeout(() => proc.exit(0), 5);
  };

  assert.equal(await shutdown(proc, {requestStop, stopTimeoutMs: 10_000}), true);
  assert.deepEqual(proc.signals, []);
});

test('a process that ignores the stop request is terminated, then killed', async () => {
  for (const [exitsOn, signals] of [
    [['SIGTERM'], ['SIGTERM']],
    [['SIGKILL'], ['SIGTERM', 'SIGKILL']],
    [[], ['SIGTERM', 'SIGKILL']],
  ]) {
    const proc = new FakeProcess(exitsOn);
    assert.equal(await shutdown(proc), exitsOn.length > 0);
    assert.deepEqual(proc.signals, signals);
  }
});

test('a stop request that fails is not waited for', async () => {
  const proc = new FakeProcess(['SIGTERM']);
  const requestStop = async () => {
    throw new Error('connect ECONNREFUSED');
  };

  const started = Date.now();
  assert.equal(await shutdown(proc, {requestStop, stopTimeoutMs: 10_000}), true);
  assert.ok(Date.now() - started < 5000);
  assert.deepEqual(proc.signals, ['SIGTERM']);
});

test('skipping the termination request goes straight to the kill', async () => {
  const proc = new FakeProcess(['SIGKILL']);

  assert.equal(await shutdown(proc, {skipTerminate: true}), true);
  assert.deepEqual(proc.signals, ['SIGKILL']);
});

test('on Windows the process is asked to close instead of being sent SIGTERM', async () => {
  const closing = new FakeProcess();
  let command;
  runCommand = (...run) => {
    command = run;
    setTimeout(() => closing.exit(0), 5);
    return 0;
  };
  assert.equal(await shutdown(closing, {platform: 'win32'}), true);
  assert.deepEqual(command, ['taskkill', ['/PID', '4242', '/T']]);
  assert.deepEqual(closing.signals, []);

  // A console program can't be asked to close, so it is killed without waiting.
  const consoleProgram = new FakeProcess(['SIGKILL']);
  runCommand = () => 1;
  const started = Date.now();
  assert.equal(await shutdown(consoleProgram, {platform: 'win32', signalTimeoutMs: 10_000}), true);
  assert.ok(Date.now() - started < 5000);
  assert.deepEqual(consoleProgram.signals, ['SIGKILL']);
});
