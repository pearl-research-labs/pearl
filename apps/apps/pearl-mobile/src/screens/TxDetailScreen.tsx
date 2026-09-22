/**
 * 交易详情页。
 */
import React from 'react';
import {ScrollView, StyleSheet, Text, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import * as Clipboard from 'expo-clipboard';
import {Alert} from 'react-native';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {ApiTxItem, grainsToPrl} from '@pearl/pearl-mobile-core';
import {colors} from '../theme';
import {Button, Card} from '../components/common';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'TxDetail'>;

export default function TxDetailScreen({route}: Props) {
  const tx: ApiTxItem = JSON.parse(route.params.txJson);
  const amount = BigInt(tx.amount);
  const incoming = tx.direction === 'in';

  const copyTxid = async () => {
    await Clipboard.setStringAsync(tx.txid);
    Alert.alert('已复制', '交易 ID 已复制到剪贴板');
  };

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={{padding: 20}}>
        <Text style={styles.title}>交易详情</Text>
        <Card>
          <Row
            label="类型"
            value={incoming ? '收款' : tx.direction === 'self' ? '内部转移' : '转账'}
          />
          <Row
            label="金额"
            value={`${incoming ? '+' : ''}${grainsToPrl(amount)} PRL`}
            tone={incoming ? 'good' : amount < 0n ? 'bad' : 'dim'}
          />
          {tx.fee ? <Row label="矿工费" value={`${grainsToPrl(BigInt(tx.fee))} PRL`} /> : null}
          <Row
            label="状态"
            value={tx.confirmations > 0 ? `${tx.confirmations} 个确认` : '等待确认'}
          />
          {tx.timestamp > 0 && (
            <Row label="时间" value={new Date(tx.timestamp * 1000).toLocaleString()} />
          )}
        </Card>
        <View style={{height: 14}} />
        <Card>
          <Text style={styles.label}>交易 ID</Text>
          <Text style={styles.txid} selectable>
            {tx.txid}
          </Text>
        </Card>
        <View style={{height: 16}} />
        <Button title="复制交易 ID" onPress={copyTxid} />
      </ScrollView>
    </SafeAreaView>
  );
}

function Row(props: {label: string; value: string; tone?: 'good' | 'bad' | 'dim'}) {
  const color =
    props.tone === 'good'
      ? colors.success
      : props.tone === 'bad'
        ? colors.danger
        : props.tone === 'dim'
          ? colors.textDim
          : colors.text;
  return (
    <View style={styles.row}>
      <Text style={styles.label}>{props.label}</Text>
      <Text style={[styles.value, {color}]}>{props.value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg},
  title: {color: colors.text, fontSize: 24, fontWeight: '700', marginBottom: 16},
  row: {flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8},
  label: {color: colors.textDim, fontSize: 14},
  value: {fontSize: 14, fontWeight: '500'},
  txid: {color: colors.text, fontSize: 13, marginTop: 6, lineHeight: 20},
});
