/**
 * 二维码扫描页：读取收款地址后返回转账页。
 */
import React, {useState} from 'react';
import {StyleSheet, Text, View} from 'react-native';
import {CameraView, useCameraPermissions} from 'expo-camera';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {Button} from '../components/common';
import {colors} from '../theme';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'Scan'>;

export default function ScanScreen({navigation, route}: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [handled, setHandled] = useState(false);

  const onScanned = ({data}: {data: string}) => {
    if (handled) return;
    setHandled(true);
    const value = data.trim();
    navigation.replace('Send', {
      toAddress: value,
      feeRate: route.params.returnFeeRate,
      balanceGrains: route.params.returnBalanceGrains,
    });
  };

  if (!permission) return <View style={styles.root} />;
  if (!permission.granted) {
    return (
      <View style={[styles.root, styles.center]}>
        <Text style={styles.note}>需要相机权限以扫描二维码</Text>
        <Button title="授权相机" onPress={requestPermission} />
        <View style={{height: 12}} />
        <Button title="返回" variant="ghost" onPress={() => navigation.goBack()} />
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <CameraView
        style={StyleSheet.absoluteFill}
        barcodeScannerSettings={{barcodeTypes: ['qr']}}
        onBarcodeScanned={handled ? undefined : onScanned}
      />
      <View style={styles.mask}>
        <View style={styles.frame} />
        <Text style={styles.note}>对准收款地址二维码</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: colors.bg},
  center: {alignItems: 'center', justifyContent: 'center', padding: 24},
  mask: {...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center'},
  frame: {
    width: 240,
    height: 240,
    borderWidth: 2,
    borderColor: colors.accent,
    borderRadius: 16,
    backgroundColor: 'transparent',
  },
  note: {color: colors.text, marginTop: 20, fontSize: 15},
});
