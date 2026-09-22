/**
 * 创建钱包：生成 12 词助记词 → 备份确认 → 完成。
 */
import React, {useEffect, useMemo, useState} from 'react';
import {Alert, ScrollView, StyleSheet, Text, TouchableOpacity, View} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';
import {createMnemonic} from '@pearl/pearl-mobile-core';
import {Button, Card} from '../components/common';
import {colors} from '../theme';
import {saveMnemonic, setSessionMnemonic} from '../wallet/secure';
import type {RootStackParamList} from '../navigation';

type Props = NativeStackScreenProps<RootStackParamList, 'CreateWallet'>;

/** 备份确认：随机抽 3 个位置让用户点选正确的词 */
function pickQuiz(words: string[]): {position: number; choices: string[]}[] {
  const positions = new Set<number>();
  while (positions.size < 3) positions.add(Math.floor(Math.random() * words.length));
  return [...positions]
    .sort((a, b) => a - b)
    .map(position => {
      const correct = words[position];
      const pool = words.filter(w => w !== correct);
      const distractors: string[] = [];
      while (distractors.length < 2) {
        const w = pool[Math.floor(Math.random() * pool.length)];
        if (!distractors.includes(w)) distractors.push(w);
      }
      const choices = [...distractors, correct].sort(() => Math.random() - 0.5);
      return {position, choices};
    });
}

export default function CreateWalletScreen({navigation}: Props) {
  const [mnemonic, setMnemonic] = useState('');
  const [step, setStep] = useState<'show' | 'quiz'>('show');
  const [quiz, setQuiz] = useState<ReturnType<typeof pickQuiz>>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [saving, setSaving] = useState(false);

  const words = useMemo(() => mnemonic.split(' ').filter(Boolean), [mnemonic]);

  useEffect(() => {
    setMnemonic(createMnemonic(12));
  }, []);

  const startQuiz = () => {
    setQuiz(pickQuiz(words));
    setAnswers({});
    setStep('quiz');
  };

  const confirm = async () => {
    for (const q of quiz) {
      if (answers[q.position] !== words[q.position]) {
        Alert.alert('备份校验失败', '选择的词与助记词不一致，请重新核对。');
        setStep('show');
        return;
      }
    }
    setSaving(true);
    try {
      await saveMnemonic(mnemonic);
      setSessionMnemonic(mnemonic);
      navigation.reset({index: 0, routes: [{name: 'Main'}]});
    } catch (e) {
      Alert.alert('保存失败', e instanceof Error ? e.message : '未知错误');
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={{padding: 20}}>
        <Text style={styles.title}>{step === 'show' ? '备份助记词' : '确认备份'}</Text>
        <Text style={styles.hint}>
          {step === 'show'
            ? '请按顺序抄下这 12 个词并妥善保管。任何得到它们的人都能取走你的资产；丢失后无法找回。'
            : '请依次点选对应位置的词，确认你已完成备份。'}
        </Text>

        {step === 'show' ? (
          <View style={styles.grid}>
            {words.map((w, i) => (
              <View key={i} style={styles.wordCell}>
                <Text style={styles.wordIndex}>{i + 1}</Text>
                <Text style={styles.wordText}>{w}</Text>
              </View>
            ))}
          </View>
        ) : (
          quiz.map(q => (
            <Card key={q.position} style={{marginBottom: 14}}>
              <Text style={styles.quizLabel}>第 {q.position + 1} 个词</Text>
              <View style={styles.choiceRow}>
                {q.choices.map(c => {
                  const active = answers[q.position] === c;
                  return (
                    <TouchableOpacity
                      key={c}
                      style={[styles.choice, active && styles.choiceActive]}
                      onPress={() => setAnswers(a => ({...a, [q.position]: c}))}
                    >
                      <Text style={[styles.choiceText, active && {color: colors.text}]}>{c}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </Card>
          ))
        )}

        <View style={{height: 16}} />
        {step === 'show' ? (
          <Button title="我已抄写，开始校验" onPress={startQuiz} />
        ) : (
          <Button
            title="完成创建"
            onPress={confirm}
            loading={saving}
            disabled={quiz.some(q => !answers[q.position])}
          />
        )}
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
  grid: {flexDirection: 'row', flexWrap: 'wrap', gap: 10},
  wordCell: {
    width: '31%',
    backgroundColor: colors.card,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    paddingVertical: 12,
    alignItems: 'center',
  },
  wordIndex: {color: colors.textDim, fontSize: 11},
  wordText: {color: colors.text, fontSize: 15, fontWeight: '600', marginTop: 2},
  quizLabel: {color: colors.textDim, fontSize: 13, marginBottom: 10},
  choiceRow: {flexDirection: 'row', gap: 10},
  choice: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    backgroundColor: colors.cardAlt,
  },
  choiceActive: {borderColor: colors.accent, backgroundColor: colors.accentDeep},
  choiceText: {color: colors.textDim, fontWeight: '600'},
});
