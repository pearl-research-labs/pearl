import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { AlertTriangle, XCircle, Shield } from 'lucide-react';
import { useState, useMemo } from 'react';
import { wordlists } from 'bip39';

type SeedVerificationProps = {
  seed: string;
  onSuccess: () => void;
  onBack: () => void;
};

const BIP39_WORDS = wordlists.english;

function pickDecoys(correct: string, count: number): string[] {
  const pool = BIP39_WORDS.filter(w => w !== correct);
  const result: string[] = [];
  const used = new Set<number>();
  while (result.length < count && used.size < pool.length) {
    const idx = Math.floor(Math.random() * pool.length);
    if (!used.has(idx)) {
      used.add(idx);
      result.push(pool[idx]);
    }
  }
  return result;
}

function shuffle<T>(arr: T[]): T[] {
  return [...arr].sort(() => Math.random() - 0.5);
}

// Pick 3 word positions to quiz: positions 3, 7, and 11 (1-indexed) for a 12-word mnemonic.
const QUIZ_POSITIONS = [2, 6, 11]; // 0-indexed → word 3, 7, 12

export default function SeedVerification({ seed, onSuccess, onBack }: SeedVerificationProps) {
  const words = seed.split(' ');

  const quizWords = QUIZ_POSITIONS.map(i => words[i] ?? '');

  const options = useMemo(
    () => quizWords.map(correct => shuffle([correct, ...pickDecoys(correct, 3)])),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [seed]
  );

  const [selected, setSelected] = useState<(string | null)[]>(quizWords.map(() => null));
  const [showError, setShowError] = useState(false);

  const handleSelect = (qIdx: number, value: string) => {
    setSelected(prev => prev.map((v, i) => (i === qIdx ? value : v)));
    setShowError(false);
  };

  const handleVerify = () => {
    if (selected.some(v => v === null)) {
      setShowError(true);
      return;
    }
    const allCorrect = selected.every((v, i) => v === quizWords[i]);
    if (allCorrect) {
      onSuccess();
    } else {
      setShowError(true);
    }
  };

  const allAnswered = selected.every(v => v !== null);

  return (
    <div className="mx-auto flex w-full max-w-md flex-col items-center">
      <div className="mb-4 rounded-2xl bg-gray-800 dark:bg-muted p-3 sm:mb-6 sm:p-4">
        <Shield className="h-8 w-8 text-white sm:h-10 sm:w-10" />
      </div>

      <h2 className="mb-2 text-center text-xl font-bold text-gray-900 dark:text-foreground sm:text-2xl">
        Verify Your Seed Phrase
      </h2>
      <p className="mb-6 text-center text-sm text-gray-600 dark:text-muted-foreground sm:text-base">
        Select the correct word for each position to confirm you've saved your seed phrase.
      </p>

      <Card className="mb-4 w-full border-gray-300 dark:border-border bg-white dark:bg-card shadow-sm sm:mb-6">
        <CardContent className="space-y-6 p-4 sm:p-6">
          {QUIZ_POSITIONS.map((pos, qIdx) => (
            <div key={pos}>
              <label className="mb-3 block text-sm font-medium text-gray-900 dark:text-foreground">
                What is word{' '}
                <span className="font-bold text-gray-900 dark:text-foreground">#{pos + 1}</span> of your seed phrase?
              </label>
              <div className="grid grid-cols-2 gap-2">
                {options[qIdx].map(word => (
                  <button
                    key={word}
                    onClick={() => handleSelect(qIdx, word)}
                    className={`rounded-lg border-2 p-3 font-mono text-sm font-medium transition-all ${
                      selected[qIdx] === word
                        ? 'border-gray-900 dark:border-foreground bg-white dark:bg-card text-gray-900 dark:text-foreground shadow-md'
                        : 'border-gray-300 dark:border-border bg-white dark:bg-card text-gray-700 dark:text-foreground/90 hover:border-gray-400 dark:hover:border-muted-foreground/60 hover:bg-gray-50 dark:hover:bg-muted/60'
                    }`}
                  >
                    {word}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {showError && (
        <div className="mb-4 w-full rounded-lg border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 p-3 shadow-sm sm:mb-6 sm:p-4">
          <div className="flex items-start gap-3">
            <XCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-600 dark:text-red-400" />
            <div className="text-sm">
              <p className="mb-1 font-medium text-red-900 dark:text-red-100">
                {!allAnswered ? 'Please answer all questions' : 'Incorrect words'}
              </p>
              <p className="text-red-700 dark:text-red-300">
                {!allAnswered
                  ? 'You need to select a word for each question to continue.'
                  : 'The words you selected do not match your seed phrase. Please go back and review it carefully.'}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="mb-4 rounded-lg border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 p-3 shadow-sm sm:mb-6 sm:p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-600 dark:text-amber-400" />
          <div className="text-sm">
            <p className="mb-1 font-medium text-amber-900 dark:text-amber-100">Why this matters:</p>
            <p className="text-amber-700 dark:text-amber-300">
              Your seed phrase is the ONLY way to recover your wallet if you lose access. We need to
              make sure you've saved it properly before continuing.
            </p>
          </div>
        </div>
      </div>

      <div className="flex w-full gap-3">
        <Button
          onClick={onBack}
          variant="outline"
          className="h-12 flex-1 border-gray-300 dark:border-border text-base shadow-sm"
        >
          Back to Seed
        </Button>
        <Button
          onClick={handleVerify}
          disabled={!allAnswered}
          className="h-12 flex-1 text-base shadow-sm"
        >
          Verify & Continue
        </Button>
      </div>
    </div>
  );
}
