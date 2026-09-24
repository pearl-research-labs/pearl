import { Card, CardContent } from '@/components/ui/card';

export default function SecurityNotice() {
  return (
    <Card className="border-gray-300 dark:border-border bg-brand-summer-sky/10 shadow-sm">
      <CardContent className="p-4">
        <div className="text-sm">
          <p className="mb-2 font-semibold text-gray-900 dark:text-foreground">Security Notice:</p>
          <ul className="space-y-1 text-xs text-gray-700 dark:text-foreground/90">
            <li>• Your password encrypts your wallet on this device</li>
            <li>• Choose a strong password you'll remember</li>
            <li>• We cannot recover your password if you forget it</li>
            <li>• You can always restore from your recovery phrase</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
