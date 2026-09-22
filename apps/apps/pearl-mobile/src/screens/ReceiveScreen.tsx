/**
 * 收款页：二维码 + 地址展示，支持复制与新地址轮换。
 */
import React, {useState} from 'react';
import {Alert, Share, StyleSheet, Text, TouchableOpacity, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import QRCode from 'react-native-qrcode-svg';
import * as Clipboard from 'expo-clipboard';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {colors} from '../theme';
import {Button, Card} from '../components/common';
import {requireMnemonic} from '../wallet/secure';
import {loadNetwork, loadAddresses, saveAddresses} from '../wallet/storage';
import {entryFor} from '@pearl/pearl-mobile-core';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'Receive'>;

export default function ReceiveScreen({route}: Props) {
  const [address, setAddress] = useState(route.params.address);
  const [rotating, setRotating] = useState(false);
  const [rotateCount, setRotateCount] = useState(0);

  const copy = async () => {
    await Clipboard.setStringAsync(address);
    Alert.alert('已复制', '地址已复制到剪贴板');
  };

  const share = () => Share.share({message: address}).catch(() => {});

  /** 轮换一个新收款地址（旧地址依然有效；每次轮换都持久化，恢复时可找回） */
  const rotate = async () => {
    setRotating(true);
    try {
      const mnemonic = await requireMnemonic();
      const network = (await loadNetwork()) ?? 'mainnet';
      const persisted = await loadAddresses(network);
      // 展示地址 = usedMax + 1 + 已轮换次数；下一个 = 再 +1
      const freshIndex = route.params.usedMaxIndex + rotateCount + 2;
      const fresh = entryFor(mnemonic, network, freshIndex, 0, false);
      const stored = {
        index: fresh.index,
        chain: fresh.chain as 0 | 1,
        address: fresh.address,
        scriptHex: fresh.scriptHex,
        used: false,
      };
      const existing = persisted?.addresses ?? [];
      await saveAddresses({
        network,
        addresses: existing.some(a => a.address === fresh.address)
          ? existing
          : [...existing, stored],
      });
      setAddress(fresh.address);
      setRotateCount(c => c + 1);
    } catch (e) {
      Alert.alert('操作失败', e instanceof Error ? e.message : '未知错误');
    } finally {
      setRotating(false);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.container}>
        <Text style={styles.title}>收款</Text>
        <Card style={styles.qrCard}>
          <QRCode value={address} size={220} backgroundColor="white" />
        </Card>
        <TouchableOpacity onPress={copy} activeOpacity={0.7}>
          <Text style={styles.address} numberOfLines={2} selectable>
            {address}
          </Text>
        </TouchableOpacity>
        <Text style={styles.hint}>点按地址可复制；旧地址长期有效</Text>
        <View style={{height: 20}} />
        <Button title="复制地址" onPress={copy} />
        <View style={{height: 12}} />
        <Button title="分享" variant="ghost" onPress={share} />
        <View style={{height: 12}} />
        <Button title="换一个新地址" variant="ghost" onPress={rotate} loading={rotating} />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg},
  container: {flex: 1, padding: 20, alignItems: 'center'},
  title: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '700',
    alignSelf: 'flex-start',
    marginBottom: 20,
  },
  qrCard: {padding: 20, backgroundColor: '#fff', borderColor: '#fff'},
  address: {
    color: colors.text,
    fontSize: 14,
    textAlign: 'center',
    marginTop: 24,
    lineHeight: 22,
    paddingHorizontal: 8,
  },
  hint: {color: colors.textDim, fontSize: 12, marginTop: 8},
});
