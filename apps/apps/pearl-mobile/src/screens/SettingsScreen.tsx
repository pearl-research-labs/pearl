/**
 * 设置页：网络切换、API 节点、备份助记词、删除钱包。
 */
import React, {useEffect, useState} from 'react';
import {Alert, ScrollView, StyleSheet, Switch, Text, TouchableOpacity, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import * as LocalAuthentication from 'expo-local-authentication';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {Button, Card, Field} from '../components/common';
import {colors} from '../theme';
import {deleteMnemonic, loadMnemonic} from '../wallet/secure';
import {
  loadNetwork,
  saveNetwork,
  saveAddresses,
  loadApiBase,
  saveApiBase,
  wipeAll,
} from '../wallet/storage';
import {getApiClient} from '../wallet/api';
import {discoverWalletAddresses} from '../wallet/engine';
import type {RootStackParamList} from '../navigation';
import type {NetworkName} from '@pearl/pearl-mobile-core';

type Props = NativeStackScreenProps<RootStackParamList, 'Settings'>;

export default function SettingsScreen({navigation}: Props) {
  const [network, setNetwork] = useState<NetworkName>('mainnet');
  const [apiBase, setApiBase] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');

  useEffect(() => {
    loadNetwork().then(n => n && setNetwork(n));
    loadApiBase().then(u => u && setApiBase(u));
  }, []);

  const switchNetwork = async (target: NetworkName) => {
    if (target === network) return;
    Alert.alert(
      '切换网络',
      `将切换到 ${target === 'mainnet' ? '主网' : '测试网'}，并重新扫描该网络上的地址。`,
      [
        {text: '取消', style: 'cancel'},
        {
          text: '切换',
          onPress: async () => {
            setBusy(true);
            try {
              const mnemonic = await loadMnemonic();
              if (!mnemonic) throw new Error('钱包不存在');
              await saveNetwork(target);
              setNetwork(target);
              const api = await getApiClient(target);
              const {addresses, nextIndex} = await discoverWalletAddresses(
                mnemonic,
                target,
                api,
                i => setProgress(`扫描地址 #${i + 1}…`)
              );
              await saveAddresses({network: target, addresses, nextIndex});
              setProgress('完成，回首页下拉刷新即可');
            } catch (e) {
              Alert.alert('切换失败', e instanceof Error ? e.message : '未知错误');
            } finally {
              setBusy(false);
            }
          },
        },
      ]
    );
  };

  const revealMnemonic = async () => {
    const ok = await LocalAuthentication.authenticateAsync({
      promptMessage: '验证身份以查看助记词',
      cancelLabel: '取消',
    });
    if (!ok.success) return;
    const m = await loadMnemonic();
    if (!m) {
      Alert.alert('未找到助记词');
      return;
    }
    Alert.alert('助记词（请勿截图）', m, [{text: '我已安全保管'}]);
  };

  const wipe = () => {
    Alert.alert('删除钱包', '将从本设备删除助记词与全部数据。若未备份助记词，资产将永久丢失！', [
      {text: '取消', style: 'cancel'},
      {
        text: '我已备份，确认删除',
        style: 'destructive',
        onPress: () =>
          Alert.alert('最后确认', '此操作不可撤销。', [
            {text: '取消', style: 'cancel'},
            {
              text: '永久删除',
              style: 'destructive',
              onPress: async () => {
                await deleteMnemonic();
                await wipeAll();
                navigation.reset({index: 0, routes: [{name: 'Welcome'}]});
              },
            },
          ]),
      },
    ]);
  };

  const saveCustomApi = async () => {
    const url = apiBase.trim().replace(/\/+$/, '');
    if (url && !/^https?:\/\//.test(url)) {
      Alert.alert('格式错误', 'API 地址需以 http(s):// 开头');
      return;
    }
    await saveApiBase(url);
    Alert.alert('已保存', url ? `自定义 API：${url}` : '已恢复默认节点');
  };

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={{padding: 20}}>
        <Text style={styles.title}>设置</Text>

        <Text style={styles.section}>网络</Text>
        <Card style={{marginBottom: 20}}>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>主网 / 测试网</Text>
            <View style={styles.rowRight}>
              <Text style={styles.rowValue}>{network === 'mainnet' ? '主网' : '测试网'}</Text>
              <Switch
                value={network === 'testnet'}
                onValueChange={v => switchNetwork(v ? 'testnet' : 'mainnet')}
                disabled={busy}
                trackColor={{true: colors.accentDeep, false: colors.border}}
                thumbColor={colors.text}
              />
            </View>
          </View>
          {!!progress && <Text style={styles.progress}>{progress}</Text>}
          <View style={{height: 12}} />
          <Field
            label="自定义数据节点（留空使用默认）"
            value={apiBase}
            onChangeText={setApiBase}
            placeholder="https://your-node:8333"
            autoCapitalize="none"
            autoCorrect={false}
          />
          <Button title="保存节点设置" variant="ghost" onPress={saveCustomApi} />
        </Card>

        <Text style={styles.section}>安全</Text>
        <Card style={{marginBottom: 20}}>
          <TouchableOpacity style={styles.linkRow} onPress={revealMnemonic}>
            <Text style={styles.linkText}>备份助记词</Text>
            <Text style={styles.linkArrow}>›</Text>
          </TouchableOpacity>
        </Card>

        <Text style={styles.section}>危险区</Text>
        <Card>
          <Button title="删除本设备上的钱包" variant="danger" onPress={wipe} />
        </Card>

        <Text style={styles.footer}>Pearl 钱包 v1.0.0 · 密钥仅存于本设备安全区</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg},
  title: {color: colors.text, fontSize: 24, fontWeight: '700', marginBottom: 16},
  section: {color: colors.textDim, fontSize: 13, marginBottom: 8, marginTop: 4},
  row: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'},
  rowLabel: {color: colors.text, fontSize: 15},
  rowRight: {flexDirection: 'row', alignItems: 'center', gap: 10},
  rowValue: {color: colors.textDim, fontSize: 14},
  progress: {color: colors.accent, fontSize: 13, marginTop: 10},
  linkRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  linkText: {color: colors.text, fontSize: 15},
  linkArrow: {color: colors.textDim, fontSize: 22},
  footer: {color: colors.textDim, fontSize: 12, textAlign: 'center', marginTop: 28},
});
