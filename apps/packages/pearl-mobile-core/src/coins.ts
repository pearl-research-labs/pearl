/**
 * 币选择：把收款金额 + 手续费映射为一组输入与输出（含找零）。
 *
 * 策略：按金额升序贪心累积（small-first），控制输入数量；
 * 找零低于粉尘阈值时并入矿工费，避免产生不可用找零输出。
 */

import {P2TR_INPUT_VBYTES, P2TR_OUTPUT_SIZE, TX_OVERHEAD_VBYTES, DUST_LIMIT} from './constants';
import {Utxo, TxOutput, estimateVBytes} from './tx';

export interface SelectionResult {
  inputs: Utxo[];
  outputs: TxOutput[];
  /** 实际手续费（Grain） */
  feeGrains: bigint;
  /** 是否有找零输出 */
  hasChange: boolean;
  /** 估算 vbytes */
  vbytes: number;
}

export interface SelectionParams {
  utxos: Utxo[];
  /** 收款输出（已按目标地址生成好 scriptPubKey） */
  recipients: TxOutput[];
  /** 费率（Grain/vbyte） */
  feeRate: number;
  /** 找零脚本（如需找零） */
  changeScript: Uint8Array;
}

export function selectCoins(params: SelectionParams): SelectionResult {
  const {utxos, recipients, feeRate, changeScript} = params;
  if (utxos.length === 0) throw new Error('没有可用余额');
  const target = recipients.reduce((sum, o) => sum + o.amountGrains, 0n);
  const rate = BigInt(Math.max(1, Math.ceil(feeRate)));

  const sorted = [...utxos].sort((a, b) =>
    a.amountGrains < b.amountGrains ? -1 : a.amountGrains > b.amountGrains ? 1 : 0
  );

  const inputs: Utxo[] = [];
  let totalIn = 0n;

  for (const u of sorted) {
    inputs.push(u);
    totalIn += u.amountGrains;

    // 先按「有找零」情形估算：in + recipients + 1 找零输出
    const vWithChange = estimateVBytes(inputs.length, recipients.length + 1);
    const feeWithChange = BigInt(vWithChange) * rate;
    const change = totalIn - target - feeWithChange;

    if (change >= BigInt(DUST_LIMIT)) {
      return {
        inputs,
        outputs: [...recipients, {scriptPubKey: changeScript, amountGrains: change}],
        feeGrains: feeWithChange,
        hasChange: true,
        vbytes: vWithChange,
      };
    }

    // 无找零情形：找零并入手续费
    const vNoChange = estimateVBytes(inputs.length, recipients.length);
    const feeNoChange = BigInt(vNoChange) * rate;
    if (totalIn >= target + feeNoChange) {
      return {
        inputs,
        outputs: [...recipients],
        feeGrains: totalIn - target,
        hasChange: false,
        vbytes: vNoChange,
      };
    }
  }

  throw new Error(`余额不足：需要 ${target} + 手续费，当前选中 ${totalIn} Grain`);
}

/** 估算单笔 1→1 转账的手续费（发送页展示用） */
export function estimateSimpleFee(utxoCount: number, feeRate: number): bigint {
  const v = estimateVBytes(Math.max(1, utxoCount), 2);
  return BigInt(v) * BigInt(Math.max(1, Math.ceil(feeRate)));
}
