/**
 * 转账页：地址（可扫码/粘贴）+ 金额 → 确认（费用明细）→ 广播。
 */
import React, {useMemo, useState} from 'react';
import {Alert, ScrollView, StyleSheet, Text, TouchableOpacity, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import * as Clipboard from 'expo-clipboard';
import {isValidAddress, prlToGrains, grainsToPrl} from '@pearl/pearl-mobile-core';
import {Button, Card, Field} from '../components/common';
import {colors} from '../theme';
import {requireMnemonic} from '../wallet/secure';
import {getApiClient} from '../wallet/api';
import {loadNetwork, loadAddresses} from '../wallet/storage';
import {PreparedSend, prepareSend} from '../wallet/engine';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'Send'>;

export default function SendScreen({navigation, route}: Props) {
  const {feeRate, balanceGrains, toAddress: presetAddress} = route.params;
  const [address, setAddress] = useState(presetAddress ?? '');
  const [amount, setAmount] = useState('');
  const [network, setNetwork] = useState<'mainnet' | 'testnet' | 'regtest'>('mainnet');
  const [busy, setBusy] = useState(false);
  const [prepared, setPrepared] = useState<PreparedSend | null>(null);

  React.useEffect(() => {
    loadNetwork().then(n => n && setNetwork(n));
  }, []);

  const addressError = useMemo(() => {
    if (!address) return '';
    return isValidAddress(address, network) ? '' : '地址无效';
  }, [address, network]);

  const amountError = useMemo(() => {
    if (!amount) return '';
    try {
      const g = prlToGrains(amount);
      if (g <= 0n) return '金额必须大于 0';
      if (g > BigInt(balanceGrains)) return '超过可用余额';
      return '';
    } catch {
      return '金额格式错误（最多 8 位小数）';
    }
  }, [amount, balanceGrains]);

  const paste = async () => {
    const text = await Clipboard.getStringAsync();
    if (text) setAddress(text.trim());
  };

  const setMax = () => {
    // 全部转出：预留费率兜底（Grain），若不足 1 Grain 则提示无法转出
    const balance = BigInt(balanceGrains);
    const reserve = BigInt(Math.max(1, feeRate)) * 1000n;
    const maxSendable = balance - reserve;
    setAmount(maxSendable > 0n ? grainsToPrl(maxSendable) : '0');
  };

  const review = async () => {
    if (!address || addressError || !amount || amountError) return;
    setBusy(true);
    try {
      const mnemonic = await requireMnemonic();
      const net = (await loadNetwork()) ?? 'mainnet';
      const persisted = await loadAddresses(net);
      if (!persisted) throw new Error('钱包数据缺失，请回到首页刷新');
      const api = await getApiClient(net);
      const send = await prepareSend({
        mnemonic,
        network: net,
        api,
        fromAddresses: persisted.addresses,
        toAddress: address,
        amountPrl: amount,
        feeRate,
      });
      setPrepared(send);
    } catch (e) {
      Alert.alert('构建交易失败', e instanceof Error ? e.message : '未知错误');
    } finally {
      setBusy(false);
    }
  };

  const doBroadcast = async () => {
    if (!prepared) return;
    setBusy(true);
    try {
      const net = (await loadNetwork()) ?? 'mainnet';
      const api = await getApiClient(net);
      const {txid} = await api.broadcast(prepared.txHex);
      Alert.alert('已广播', `交易 ID:\n${txid}`, [
        {text: '完成', onPress: () => navigation.popToTop()},
      ]);
    } catch (e) {
      Alert.alert('广播失败', e instanceof Error ? e.message : '未知错误');
    } finally {
      setBusy(false);
    }
  };

  if (prepared) {
    return (
      <SafeAreaView style={styles.root}>
        <ScrollView contentContainerStyle={{padding: 20}}>
          <Text style={styles.title}>确认转账</Text>
          <Card>
            <Row label="收款地址" value={prepared.toAddress} mono />
            <Row label="金额" value={`${grainsToPrl(prepared.amountGrains)} PRL`} />
            <Row label="矿工费" value={`${grainsToPrl(prepared.feeGrains)} PRL`} />
            <Row
              label="合计支出"
              value={`${grainsToPrl(prepared.amountGrains + prepared.feeGrains)} PRL`}
              strong
            />
            <Row label="交易大小" value={`约 ${prepared.vbytes} vB`} />
          </Card>
          <Text style={styles.warn}>请仔细核对地址与金额，交易一经确认不可撤销。</Text>
          <View style={{height: 16}} />
          <Button title="确认并广播" onPress={doBroadcast} loading={busy} />
          <View style={{height: 12}} />
          <Button
            title="返回修改"
            variant="ghost"
            onPress={() => setPrepared(null)}
            disabled={busy}
          />
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={{padding: 20}} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>转账</Text>
        <Card style={{marginBottom: 16}}>
          <Field
            label="收款地址"
            value={address}
            onChangeText={setAddress}
            placeholder="prl1p…"
            autoCapitalize="none"
            autoCorrect={false}
            error={addressError}
          />
          <View style={styles.toolRow}>
            <ToolButton label="粘贴" onPress={paste} />
            <ToolButton
              label="扫码"
              onPress={() =>
                navigation.navigate('Scan', {
                  returnFeeRate: feeRate,
                  returnBalanceGrains: balanceGrains,
                })
              }
            />
          </View>
          <View style={{height: 16}} />
          <Field
            label={`金额（PRL）· 可用 ${grainsToPrl(BigInt(balanceGrains))}`}
            value={amount}
            onChangeText={setAmount}
            placeholder="0.00"
            keyboardType="decimal-pad"
            error={amountError}
          />
          <View style={styles.toolRow}>
            <ToolButton label="全部转出" onPress={setMax} />
          </View>
          <Text style={styles.feeNote}>当前费率约 {feeRate} Grain/vB，手续费在确认页展示</Text>
        </Card>
        <Button
          title="预览交易"
          onPress={review}
          loading={busy}
          disabled={!address || !!addressError || !amount || !!amountError}
        />
      </ScrollView>
    </SafeAreaView>
  );
}

function Row(props: {label: string; value: string; mono?: boolean; strong?: boolean}) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{props.label}</Text>
      <Text
        style={[
          styles.rowValue,
          props.mono && {fontSize: 12},
          props.strong && {color: colors.text, fontWeight: '700'},
        ]}
        numberOfLines={props.mono ? 2 : 1}
        selectable={props.mono}
      >
        {props.value}
      </Text>
    </View>
  );
}

function ToolButton(props: {label: string; onPress: () => void}) {
  return (
    <TouchableOpacity style={styles.tool} onPress={props.onPress}>
      <Text style={styles.toolText}>{props.label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg},
  title: {color: colors.text, fontSize: 24, fontWeight: '700', marginBottom: 16},
  toolRow: {flexDirection: 'row', gap: 10},
  tool: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.cardAlt,
  },
  toolText: {color: colors.accent, fontSize: 13, fontWeight: '600'},
  feeNote: {color: colors.textDim, fontSize: 12, marginTop: 14},
  row: {flexDirection: 'row', justifyContent: 'space-between', gap: 12, paddingVertical: 8},
  rowLabel: {color: colors.textDim, fontSize: 14},
  rowValue: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '500',
    flexShrink: 1,
    textAlign: 'right',
  },
  warn: {color: colors.warning, fontSize: 13, marginTop: 16, lineHeight: 18},
});
