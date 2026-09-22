/**
 * 欢迎页：创建新钱包或导入已有助记词。
 */
import React from 'react';
import {Image, StyleSheet, Text, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {Button} from '../components/common';
import {colors} from '../theme';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'Welcome'>;

export default function WelcomeScreen({navigation}: Props) {
  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.center}>
        <Image source={require('../../assets/icon.png')} style={styles.logo} />
        <Text style={styles.title}>Pearl 钱包</Text>
        <Text style={styles.subtitle}>
          你的密钥，只存在这台设备上。{'\n'}面向 Pearl 网络的轻量移动钱包。
        </Text>
      </View>
      <View style={styles.footer}>
        <Button title="创建新钱包" onPress={() => navigation.navigate('CreateWallet')} />
        <View style={{height: 12}} />
        <Button
          title="导入已有助记词"
          variant="ghost"
          onPress={() => navigation.navigate('ImportWallet')}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg, padding: 24},
  center: {flex: 1, alignItems: 'center', justifyContent: 'center'},
  logo: {width: 96, height: 96, borderRadius: 24, marginBottom: 20},
  title: {color: colors.text, fontSize: 30, fontWeight: '700'},
  subtitle: {
    color: colors.textDim,
    fontSize: 15,
    textAlign: 'center',
    marginTop: 12,
    lineHeight: 22,
  },
  footer: {paddingBottom: 8},
});
