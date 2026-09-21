import {useState} from 'react';
import {Input} from '@/components/ui/input';
import {Label} from '@/components/ui/label';
import {Button} from '@/components/ui/button';
import {Eye, EyeOff} from 'lucide-react';

type PasswordFieldProps = {
  onChange: (v: string) => void;
};

export default function PasswordField({onChange}: PasswordFieldProps) {
  const [showPassword, setShowPassword] = useState(false);
  return (
    <div className="space-y-2">
      <Label htmlFor="password" className="text-gray-900 dark:text-foreground">
        Password
      </Label>
      <div className="relative">
        <Input
          type={showPassword ? 'text' : 'password'}
          onChange={e => onChange(e.target.value)}
          placeholder="Enter a strong password"
          className="focus:border-brand-green focus:ring-brand-green/20 border-gray-300 dark:border-border bg-white dark:bg-card pr-10 text-gray-900 dark:text-foreground shadow-sm focus:ring-2"
        />
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="absolute right-1 top-1 h-8 w-8 text-gray-500 dark:text-muted-foreground hover:bg-gray-100 dark:hover:bg-muted hover:text-gray-700 dark:hover:text-foreground"
          onClick={() => setShowPassword(!showPassword)}
        >
          {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
        </Button>
      </div>
    </div>
  );
}
