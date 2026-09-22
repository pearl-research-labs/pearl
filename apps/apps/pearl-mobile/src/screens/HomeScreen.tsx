/**
 * 首页：余额总览 + 最近交易 + 收/发入口，下拉刷新。
 */
import React, {useCallback, useEffect, useState} from 'react';
import {FlatList, RefreshControl, StyleSheet, Text, TouchableOpacity, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {ApiTxItem, grainsToPrl} from '@pearl/pearl-mobile-core';
import {colors} from '../theme';
import {Card} from '../components/common';
import {getApiClient} from '../wallet/api';
import {WalletSnapshot, refreshSnapshot, loadHistory} from '../wallet/engine';
import {requireMnemonic} from '../wallet/secure';
import {
  loadNetwork,
  loadAddresses,
  saveAddresses,
  saveHistoryCache,
  loadHistoryCache,
} from '../wallet/storage';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'Main'>;

export default function HomeScreen({navigation}: Props) {
  const [snapshot, setSnapshot] = useState<WalletSnapshot | null>(null);
  const [history, setHistory] = useState<ApiTxItem[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setError('');
    try {
      const mnemonic = await requireMnemonic();
      const network = (await loadNetwork()) ?? 'mainnet';
      const api = await getApiClient(network);

      const persisted = await loadAddresses(network);
      // 轮换持久化地址发现：从最大已用 index 之后继续探测，
      // 兼顾日常刷新与恢复导入（发现到的地址会持久化）
      const snap = await refreshSnapshot(mnemonic, network, api, persisted?.addresses ?? []);
      setSnapshot(snap);
      await saveAddresses({network, addresses: snap.addresses});

      const txs = await loadHistory(api, snap.addresses, snap.changeAddress);
      setHistory(txs);
      await saveHistoryCache(network, txs);
    } catch (e) {
      setError(e instanceof Error ? e.message : '刷新失败');
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadNetwork().then(async network => {
      if (!network) return;
      const cached = await loadHistoryCache(network);
      setHistory(cached);
    });
    refresh();
  }, [refresh]);

  const confirmed = snapshot ? grainsToPrl(snapshot.confirmed) : '—';
  const unconfirmed = snapshot?.unconfirmed ?? 0n;

  return (
    <SafeAreaView style={styles.root}>
      <FlatList
        data={history}
        keyExtractor={item => item.txid}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={refresh} tintColor={colors.accent} />
        }
        ListHeaderComponent={
          <View>
            <View style={styles.headerRow}>
              <Text style={styles.headerTitle}>Pearl 钱包</Text>
              <TouchableOpacity onPress={() => navigation.navigate('Settings')}>
                <Text style={styles.headerGear}>⚙️</Text>
              </TouchableOpacity>
            </View>

            <Card style={styles.balanceCard}>
              <Text style={styles.balanceLabel}>可用余额</Text>
              <Text style={styles.balanceValue}>
                {confirmed} <Text style={styles.balanceUnit}>PRL</Text>
              </Text>
              {unconfirmed !== 0n && (
                <Text style={styles.unconfirmed}>未确认：{grainsToPrl(unconfirmed)} PRL</Text>
              )}
              {snapshot ? (
                <Text style={styles.heightText}>区块高度 {snapshot.blockHeight}</Text>
              ) : null}
              <View style={styles.actionRow}>
                <TouchableOpacity
                  style={styles.actionButton}
                  onPress={() =>
                    snapshot &&
                    navigation.navigate('Receive', {
                      address: snapshot.receiveAddress,
                      usedMaxIndex: snapshot.usedMaxIndex,
                    })
                  }
                >
                  <Text style={styles.actionText}>收款</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.actionButton, styles.actionPrimary]}
                  onPress={() =>
                    snapshot &&
                    navigation.navigate('Send', {
                      feeRate: snapshot.feeRate,
                      balanceGrains: snapshot.confirmed.toString(),
                    })
                  }
                >
                  <Text style={styles.actionText}>转账</Text>
                </TouchableOpacity>
              </View>
            </Card>

            {!!error && <Text style={styles.error}>{error}</Text>}
            <Text style={styles.sectionTitle}>交易记录</Text>
          </View>
        }
        ListEmptyComponent={
          <Text style={styles.empty}>{refreshing ? '加载中…' : '暂无交易记录'}</Text>
        }
        renderItem={({item}) => (
          <TxRow
            item={item}
            onPress={() => navigation.navigate('TxDetail', {txJson: JSON.stringify(item)})}
          />
        )}
        contentContainerStyle={{padding: 20}}
      />
    </SafeAreaView>
  );
}

function TxRow({item, onPress}: {item: ApiTxItem; onPress?: () => void}) {
  const positive = item.direction === 'in';
  const self = item.direction === 'self';
  const amount = BigInt(item.amount);
  const shown = self ? '0' : grainsToPrl(positive ? amount : -amount);
  const date = item.timestamp ? new Date(item.timestamp * 1000).toLocaleString() : '未确认';
  return (
    <TouchableOpacity style={styles.txRow} onPress={onPress} activeOpacity={0.7}>
      <View style={[styles.txIcon, {backgroundColor: positive ? '#12392E' : '#3A1A22'}]}>
        <Text style={{fontSize: 16}}>{positive ? '⬇' : self ? '↻' : '⬆'}</Text>
      </View>
      <View style={{flex: 1}}>
        <Text style={styles.txId} numberOfLines={1}>
          {item.txid}
        </Text>
        <Text style={styles.txDate}>
          {date} · {item.confirmations > 0 ? `${item.confirmations} 确认` : '待确认'}
        </Text>
      </View>
      <Text
        style={[
          styles.txAmount,
          {color: positive ? colors.success : self ? colors.textDim : colors.danger},
        ]}
      >
        {positive ? '+' : ''}
        {shown} PRL
      </Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg},
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  headerTitle: {color: colors.text, fontSize: 22, fontWeight: '700'},
  headerGear: {fontSize: 22},
  balanceCard: {marginBottom: 20},
  balanceLabel: {color: colors.textDim, fontSize: 13},
  balanceValue: {color: colors.text, fontSize: 34, fontWeight: '700', marginTop: 6},
  balanceUnit: {fontSize: 16, color: colors.textDim},
  unconfirmed: {color: colors.warning, fontSize: 13, marginTop: 4},
  heightText: {color: colors.textDim, fontSize: 12, marginTop: 8},
  actionRow: {flexDirection: 'row', gap: 12, marginTop: 16},
  actionButton: {
    flex: 1,
    height: 46,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.cardAlt,
    borderWidth: 1,
    borderColor: colors.border,
  },
  actionPrimary: {backgroundColor: colors.accentDeep, borderColor: colors.accentDeep},
  actionText: {color: colors.text, fontSize: 16, fontWeight: '600'},
  sectionTitle: {color: colors.text, fontSize: 17, fontWeight: '600', marginBottom: 8},
  empty: {color: colors.textDim, textAlign: 'center', marginTop: 32},
  error: {color: colors.danger, marginBottom: 12},
  txRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  txIcon: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
  },
  txId: {color: colors.text, fontSize: 14, fontWeight: '500'},
  txDate: {color: colors.textDim, fontSize: 12, marginTop: 2},
  txAmount: {fontSize: 15, fontWeight: '600'},
});
