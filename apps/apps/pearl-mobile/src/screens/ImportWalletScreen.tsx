/**
 * 导入钱包：粘贴/输入助记词 → 地址发现（含进度）→ 完成。
 */
import React, {useState} from 'react';
import {Alert, ScrollView, StyleSheet, Text, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {isValidMnemonic} from '@pearl/pearl-mobile-core';
import {Button, Card, Field} from '../components/common';
import {colors} from '../theme';
import {saveMnemonic, deleteMnemonic, setSessionMnemonic} from '../wallet/secure';
import {getApiClient} from '../wallet/api';
import {discoverWalletAddresses} from '../wallet/engine';
import {loadNetwork, saveAddresses, saveNetwork} from '../wallet/storage';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'ImportWallet'>;

export default function ImportWalletScreen({navigation}: Props) {
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState('');

  const doImport = async () => {
    const mnemonic = text.trim().toLowerCase().split(/\s+/).join(' ');
    if (!isValidMnemonic(mnemonic)) {
      setError('助记词无效：请检查单词拼写、顺序与数量（12/24 词）');
      return;
    }
    setError('');
    setScanning(true);
    try {
      await saveMnemonic(mnemonic);
      setSessionMnemonic(mnemonic);

      const network = (await loadNetwork()) ?? 'mainnet';
      await saveNetwork(network);
      const api = await getApiClient(network);

      // gap-limit 扫描，恢复历史地址（轮换未使用地址由首页快照按需补齐）
      const addresses = await discoverWalletAddresses(mnemonic, network, api, i =>
        setProgress(`正在扫描地址 #${i + 1}…`)
      );
      await saveAddresses({network, addresses});

      navigation.reset({index: 0, routes: [{name: 'Main'}]});
    } catch (e) {
      // 导入中途失败（如网络错误导致扫描中断）时回滚：
      // 删除已写入的助记词，避免留下「有钱包、无地址簿」的残缺状态
      await deleteMnemonic().catch(() => {});
      setSessionMnemonic(null);
      Alert.alert(
        '导入失败',
        (e instanceof Error ? e.message : '未知错误') + '\n\n已回滚，可重试导入。'
      );
    } finally {
      setScanning(false);
      setProgress('');
    }
  };

  const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={{padding: 20}} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>导入钱包</Text>
        <Text style={styles.hint}>
          输入你的 12 或 24 个助记词（以空格分隔）。导入过程会在链上扫描使用过的地址，需要一些时间。
        </Text>
        <Card>
          <Field
            label={`助记词（当前 ${wordCount} 词）`}
            value={text}
            onChangeText={setText}
            placeholder="word1 word2 word3 …"
            multiline
            numberOfLines={4}
            autoCapitalize="none"
            autoCorrect={false}
            style={{minHeight: 96, textAlignVertical: 'top'}}
            error={error}
          />
        </Card>
        {scanning && (
          <View style={styles.progressWrap}>
            <Text style={styles.progress}>{progress || '正在保存…'}</Text>
          </View>
        )}
        <View style={{height: 16}} />
        <Button
          title="开始导入"
          onPress={doImport}
          loading={scanning}
          disabled={wordCount !== 12 && wordCount !== 24}
        />
        <View style={{height: 12}} />
        <Button title="返回" variant="ghost" onPress={() => navigation.goBack()} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg},
  title: {color: colors.text, fontSize: 24, fontWeight: '700', marginBottom: 8},
  hint: {color: colors.textDim, fontSize: 14, lineHeight: 20, marginBottom: 20},
  progressWrap: {marginTop: 16, alignItems: 'center'},
  progress: {color: colors.accent, fontSize: 14},
});
