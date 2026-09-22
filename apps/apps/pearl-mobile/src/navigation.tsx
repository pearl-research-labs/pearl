/**
 * 导航结构定义与导航器。
 */
import React from 'react';
import {DarkTheme, NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {colors} from './theme';
import WelcomeScreen from './screens/WelcomeScreen';
import CreateWalletScreen from './screens/CreateWalletScreen';
import ImportWalletScreen from './screens/ImportWalletScreen';
import HomeScreen from './screens/HomeScreen';
import ReceiveScreen from './screens/ReceiveScreen';
import SendScreen from './screens/SendScreen';
import ScanScreen from './screens/ScanScreen';
import TxDetailScreen from './screens/TxDetailScreen';
import SettingsScreen from './screens/SettingsScreen';

export type RootStackParamList = {
  Welcome: undefined;
  CreateWallet: undefined;
  ImportWallet: undefined;
  Main: undefined;
  Receive: {address: string; usedMaxIndex: number};
  Send: {feeRate: number; balanceGrains: string; toAddress?: string};
  Scan: {returnFeeRate: number; returnBalanceGrains: string};
  TxDetail: {txJson: string};
  Settings: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const navTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: colors.bg,
    card: colors.bg,
    text: colors.text,
    border: colors.border,
    primary: colors.accent,
  },
};

export default function RootNavigator(props: {hasWallet: boolean}) {
  return (
    <NavigationContainer theme={navTheme}>
      <Stack.Navigator
        initialRouteName={props.hasWallet ? 'Main' : 'Welcome'}
        screenOptions={{
          headerStyle: {backgroundColor: colors.bg},
          headerTintColor: colors.text,
          headerShadowVisible: false,
          contentStyle: {backgroundColor: colors.bg},
        }}
      >
        <Stack.Screen name="Welcome" component={WelcomeScreen} options={{headerShown: false}} />
        <Stack.Screen
          name="CreateWallet"
          component={CreateWalletScreen}
          options={{title: '创建钱包'}}
        />
        <Stack.Screen
          name="ImportWallet"
          component={ImportWalletScreen}
          options={{title: '导入钱包'}}
        />
        <Stack.Screen name="Main" component={HomeScreen} options={{headerShown: false}} />
        <Stack.Screen name="Receive" component={ReceiveScreen} options={{title: '收款'}} />
        <Stack.Screen name="Send" component={SendScreen} options={{title: '转账'}} />
        <Stack.Screen
          name="Scan"
          component={ScanScreen}
          options={{title: '扫码', headerTransparent: true}}
        />
        <Stack.Screen name="TxDetail" component={TxDetailScreen} options={{title: '交易详情'}} />
        <Stack.Screen name="Settings" component={SettingsScreen} options={{title: '设置'}} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
