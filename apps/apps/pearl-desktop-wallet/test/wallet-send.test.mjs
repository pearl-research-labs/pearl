import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function loadSource(mocks) {
  const modules = new Map();
  const load = filename => {
    if (modules.has(filename)) return modules.get(filename).exports;
    const module = {exports: {}};
    modules.set(filename, module);
    const {outputText} = ts.transpileModule(readFileSync(filename, 'utf8'), {
      compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true},
    });
    const require = request => {
      if (Object.hasOwn(mocks, request)) return mocks[request];
      if (request.startsWith('.')) return load(resolve(dirname(filename), `${request}.ts`));
      return createRequire(filename)(request);
    };
    new Function('require', 'module', 'exports', outputText)(require, module, module.exports);
    return module.exports;
  };
  return path => load(resolve(appDir, path));
}

function walletBridge(reply) {
  const handlers = new Map();
  const requests = [];
  let bridge;
  const load = loadSource({
    axios: {
      post: async (_url, request) => {
        requests.push(request);
        if (reply instanceof Error) throw reply;
        return {data: reply};
      },
      isAxiosError: () => false,
    },
    electron: {
      ipcMain: {handle: (channel, handler) => handlers.set(channel, handler)},
      ipcRenderer: {
        invoke: async (channel, ...args) => {
          try {
            return structuredClone(await handlers.get(channel)({}, ...structuredClone(args)));
          } catch (error) {
            // Electron drops custom properties on errors thrown by an IPC handler.
            throw new Error(error.message);
          }
        },
      },
      contextBridge: {
        exposeInMainWorld: (_name, value) => {
          bridge = value;
        },
      },
    },
    '../clients/blockbook-client': {BlockbookClient: {}},
  });
  const {RpcClient} = load('src/main/services/rpc-client.ts');
  const {WalletRpcMethods} = load('src/main/services/wallet-service/wallet-rpc-methods.ts');
  const {registerWalletIpc} = load('src/main/ipc/register-wallet-ipc.ts');
  const rpc = new RpcClient({rpcHost: 'http://127.0.0.1', rpcPort: 0, rpcUser: 'test', rpcPassword: 'test'});
  registerWalletIpc({ensureWalletService: () => new WalletRpcMethods(rpc)});
  load('src/preload/index.ts');
  const {getSendFailure} = load('src/renderer/src/lib/send-result.ts');
  return {bridge, requests, getSendFailure};
}

test('fresh-send retry requires numeric RPC code -9 across the RPC, IPC and preload path', async t => {
  const safeMessage =
    'No peers are connected. Nothing was signed or sent and your funds are untouched. ' +
    'Check your connection and try again.';
  const cases = [
    {name: 'preflight with unrelated wording', error: {code: -9, message: 'offline'}, canRetry: true},
    {name: 'peer text lookalike', error: {code: -32603, message: 'peer says not relayed'}, canRetry: false},
    {name: 'exact warning lookalike', error: {code: -32603, message: safeMessage}, canRetry: false},
    {name: 'message-only old daemon', error: {message: 'not relayed'}, canRetry: false},
    {name: 'code in message', error: {code: -32603, message: '{"code":-9}: not relayed'}, canRetry: false},
    {name: 'string code', error: {code: '-9', message: 'not relayed'}, canRetry: false},
    {name: 'transport failure', transport: new Error('not relayed'), canRetry: false},
    {name: 'non-RPC error code', transport: Object.assign(new Error('not relayed'), {code: -9}), canRetry: false},
  ];
  for (const scenario of cases) {
    await t.test(scenario.name, async () => {
      const {bridge, getSendFailure} = walletBridge(scenario.transport ?? {error: scenario.error});
      const result = structuredClone(await bridge.wallet.sendFromDefaultAccount('recipient', 1, 0.00001));
      assert.equal(result.ok, false);
      const failure = getSendFailure(result.error);
      assert.equal(failure.canRetry, scenario.canRetry);
      assert.equal(failure.message, scenario.canRetry ? safeMessage : (scenario.error ?? scenario.transport).message);
      assert.equal(result.error.code === -9, scenario.canRetry);
    });
  }
});

test('successful sends preserve the txid and send arguments through IPC and preload', async () => {
  const txid = 'ab'.repeat(32);
  const {bridge, requests} = walletBridge({result: txid, error: null});
  assert.deepEqual(await bridge.wallet.sendFromDefaultAccount('recipient', 1, 0.00001), {ok: true, txid});
  assert.equal(requests[0].method, 'sendmany');
  assert.deepEqual(requests[0].params, ['default', {recipient: 1}, 0.00001, 0]);
});

test('rebroadcast failure stays generic even when its message contains not relayed', async () => {
  const {bridge} = walletBridge({error: {code: -32603, message: 'not relayed'}});
  await assert.rejects(bridge.wallet.rebroadcastTransaction('ab'.repeat(32)), error => {
    assert.equal(error.message, 'not relayed');
    assert.equal(error.code, undefined);
    return true;
  });
});
