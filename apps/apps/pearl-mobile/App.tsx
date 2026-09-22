/**
 * 应用入口：启动时判定是否已有钱包；监听前后台切换以清理内存中的助记词。
 */
import React, {useEffect, useState} from 'react';
import {AppState, StatusBar, View} from 'react-native';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import RootNavigator from './src/navigation';
import {hasMnemonic, setSessionMnemonic} from './src/wallet/secure';
import {colors} from './src/theme';

export default function App() {
  const [ready, setReady] = useState(false);
  const [hasWallet, setHasWallet] = useState(false);

  useEffect(() => {
    hasMnemonic()
      .then(setHasWallet)
      .finally(() => setReady(true));

    // 应用退到后台即清除内存中的助记词（安全存储中的副本不受影响）
    const sub = AppState.addEventListener('change', state => {
      if (state !== 'active') setSessionMnemonic(null);
    });
    return () => sub.remove();
  }, []);

  if (!ready) {
    return <View style={{flex: 1, backgroundColor: colors.bg}} />;
  }

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor={colors.bg} />
      <RootNavigator hasWallet={hasWallet} />
    </SafeAreaProvider>
  );
}
