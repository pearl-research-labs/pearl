import React from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  TextInputProps,
  ViewStyle,
} from 'react-native';
import {colors, radius} from '../theme';

export function Button(props: {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'ghost' | 'danger';
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle;
}) {
  const {title, onPress, variant = 'primary', disabled, loading, style} = props;
  const bg =
    variant === 'primary'
      ? colors.accentDeep
      : variant === 'danger'
        ? colors.danger
        : 'transparent';
  return (
    <TouchableOpacity
      style={[
        styles.button,
        {backgroundColor: bg},
        variant === 'ghost' && styles.buttonGhost,
        (disabled || loading) && styles.buttonDisabled,
        style,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.8}
    >
      {loading ? (
        <ActivityIndicator color={colors.text} />
      ) : (
        <Text style={[styles.buttonText, variant === 'ghost' && {color: colors.accent}]}>
          {title}
        </Text>
      )}
    </TouchableOpacity>
  );
}

export function Field(props: TextInputProps & {label: string; error?: string}) {
  const {label, error, ...rest} = props;
  return (
    <View style={styles.fieldWrap}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput
        placeholderTextColor={colors.textDim}
        style={[styles.input, !!error && {borderColor: colors.danger}]}
        {...rest}
      />
      {!!error && <Text style={styles.fieldError}>{error}</Text>}
    </View>
  );
}

export function Card(props: {children: React.ReactNode; style?: ViewStyle}) {
  return <View style={[styles.card, props.style]}>{props.children}</View>;
}

const styles = StyleSheet.create({
  button: {
    height: 52,
    borderRadius: radius.button,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  buttonGhost: {
    borderWidth: 1,
    borderColor: colors.border,
  },
  buttonDisabled: {opacity: 0.5},
  buttonText: {color: colors.text, fontSize: 16, fontWeight: '600'},
  fieldWrap: {marginBottom: 16},
  fieldLabel: {color: colors.textDim, fontSize: 13, marginBottom: 6},
  input: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.button,
    color: colors.text,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
  },
  fieldError: {color: colors.danger, fontSize: 12, marginTop: 4},
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.card,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 16,
  },
});
