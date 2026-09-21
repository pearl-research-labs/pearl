import { zxcvbn, zxcvbnOptions } from '@zxcvbn-ts/core';
import * as zxcvbnCommonPackage from '@zxcvbn-ts/language-common';
import * as zxcvbnEnPackage from '@zxcvbn-ts/language-en';

zxcvbnOptions.setOptions({
  translations: zxcvbnEnPackage.translations,
  graphs: zxcvbnCommonPackage.adjacencyGraphs,
  dictionary: {
    ...zxcvbnCommonPackage.dictionary,
    ...zxcvbnEnPackage.dictionary,
  },
});

type PasswordStrengthProps = {
  password: string;
};

const strengthConfig = {
  0: { label: 'Very weak', color: 'bg-red-500', textColor: 'text-red-600 dark:text-red-400', width: 'w-1/5' },
  1: { label: 'Weak', color: 'bg-orange-500', textColor: 'text-orange-600 dark:text-orange-400', width: 'w-2/5' },
  2: { label: 'Fair', color: 'bg-yellow-500', textColor: 'text-yellow-600 dark:text-yellow-400', width: 'w-3/5' },
  3: { label: 'Strong', color: 'bg-green-500', textColor: 'text-green-600 dark:text-green-400', width: 'w-4/5' },
  4: { label: 'Very strong', color: 'bg-green-600', textColor: 'text-green-700 dark:text-green-300', width: 'w-full' },
} as const;

export default function PasswordStrength({ password }: PasswordStrengthProps) {
  if (!password) return null;

  const result = zxcvbn(password);
  const config = strengthConfig[result.score];

  const warning = result.feedback.warning;
  const suggestions = result.feedback.suggestions;

  return (
    <div className="space-y-2">
      <div className="h-2 w-full bg-gray-200 dark:bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${config.color} transition-all duration-300 ease-in-out ${config.width}`}
        />
      </div>

      <div className="flex items-center justify-between text-xs">
        <span className={`font-medium ${config.textColor}`}>
          {config.label}
        </span>
        <span className="text-gray-500 dark:text-muted-foreground">
          {password.length} character{password.length !== 1 ? 's' : ''}
        </span>
      </div>

      {warning && (
        <p className="text-xs text-amber-700 dark:text-amber-300">{warning}</p>
      )}

      {suggestions.length > 0 && (
        <ul className="text-xs text-gray-500 dark:text-muted-foreground space-y-0.5">
          {suggestions.map((s, i) => (
            <li key={i}>• {s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
