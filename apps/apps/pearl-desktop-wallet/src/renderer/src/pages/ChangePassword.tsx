import { useState } from 'react';
import { ArrowLeft, Key, Eye, EyeOff, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useWalletStore } from '../store/walletStore';
import PasswordStrength from './create-wallet/PasswordStrength';

export default function ChangePassword() {
  const navigate = useNavigate();
  const { walletName } = useWalletStore();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isChanging, setIsChanging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const validateInputs = (): boolean => {
    setError(null);

    if (!currentPassword.trim()) {
      setError('Please enter your current password');
      return false;
    }

    if (!newPassword.trim()) {
      setError('Please enter a new password');
      return false;
    }

    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return false;
    }

    if (currentPassword === newPassword) {
      setError('New password must be different from current password');
      return false;
    }

    return true;
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateInputs()) return;

    setIsChanging(true);
    setError(null);
    setSuccess(null);

    console.log('🔄 Changing wallet password...');

    try {
      await window.appBridge.wallet.changeWalletPassphrase(currentPassword, newPassword);

      console.log('✅ Password changed successfully');
      setSuccess('Password changed successfully!');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      console.error('❌ Exception during password change:', err);
      setError('An error occurred while changing the password');
    } finally {
      setIsChanging(false);
    }
  };

  return (
    <div className="flex h-full w-full flex-col bg-transparent">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center gap-4 border-b border-gray-200 dark:border-border bg-white/80 dark:bg-card/80 p-6 shadow-sm backdrop-blur-sm">
        <button
          onClick={() => navigate('/wallet')}
          className="rounded-lg p-2 transition-colors hover:bg-gray-100 dark:hover:bg-muted"
        >
          <ArrowLeft className="h-5 w-5 text-gray-700 dark:text-foreground/90" />
        </button>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-foreground">Change Password</h1>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-8 py-12">
        <div className="mx-auto flex max-w-md flex-col items-center">
          {/* Header Info */}
          <div className="mb-8 text-center">
            <div className="mb-6 inline-flex h-16 w-16 items-center justify-center rounded-full bg-amber-600">
              <Key className="h-8 w-8 text-white" />
            </div>
            <h2 className="mb-2 text-2xl font-bold text-gray-900 dark:text-foreground">Change Wallet Password</h2>
            <p className="text-gray-600 dark:text-muted-foreground">Update the password for wallet "{walletName}"</p>
          </div>

          {/* Change Password Form */}
          <form onSubmit={handleChangePassword} className="w-full space-y-6">
            {/* Current Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 dark:text-foreground/90">Current Password</label>
              <div className="relative">
                <input
                  type={showCurrentPassword ? 'text' : 'password'}
                  value={currentPassword}
                  onChange={e => setCurrentPassword(e.target.value)}
                  placeholder="Enter your current password"
                  className="focus:border-brand-green focus:ring-brand-green/20 w-full rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card px-4 py-3 pr-12 text-gray-900 dark:text-foreground placeholder-gray-400 dark:placeholder-muted-foreground/70 focus:outline-none focus:ring-2"
                  disabled={isChanging}
                />
                <button
                  type="button"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-500 dark:text-muted-foreground transition-colors hover:text-gray-700 dark:hover:text-foreground"
                  disabled={isChanging}
                >
                  {showCurrentPassword ? (
                    <EyeOff className="h-5 w-5" />
                  ) : (
                    <Eye className="h-5 w-5" />
                  )}
                </button>
              </div>
            </div>

            {/* New Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 dark:text-foreground/90">New Password</label>
              <div className="relative">
                <input
                  type={showNewPassword ? 'text' : 'password'}
                  value={newPassword}
                  onChange={e => setNewPassword(e.target.value)}
                  placeholder="Enter a new secure password"
                  className="focus:border-brand-green focus:ring-brand-green/20 w-full rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card px-4 py-3 pr-12 text-gray-900 dark:text-foreground placeholder-gray-400 dark:placeholder-muted-foreground/70 focus:outline-none focus:ring-2"
                  disabled={isChanging}
                />
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-500 dark:text-muted-foreground transition-colors hover:text-gray-700 dark:hover:text-foreground"
                  disabled={isChanging}
                >
                  {showNewPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {newPassword && (
                <div className="mt-3">
                  <PasswordStrength password={newPassword} />
                </div>
              )}
            </div>

            {/* Confirm New Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 dark:text-foreground/90">Confirm New Password</label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  placeholder="Confirm your new password"
                  className="focus:border-brand-green focus:ring-brand-green/20 w-full rounded-lg border border-gray-300 dark:border-border bg-white dark:bg-card px-4 py-3 pr-12 text-gray-900 dark:text-foreground placeholder-gray-400 dark:placeholder-muted-foreground/70 focus:outline-none focus:ring-2"
                  disabled={isChanging}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-500 dark:text-muted-foreground transition-colors hover:text-gray-700 dark:hover:text-foreground"
                  disabled={isChanging}
                >
                  {showConfirmPassword ? (
                    <EyeOff className="h-5 w-5" />
                  ) : (
                    <Eye className="h-5 w-5" />
                  )}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center gap-3 rounded-lg border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 p-4">
                <AlertCircle className="h-5 w-5 flex-shrink-0 text-red-600 dark:text-red-400" />
                <span className="text-red-700 dark:text-red-300">{error}</span>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="flex items-center gap-3 rounded-lg border border-green-200 dark:border-green-900 bg-green-50 dark:bg-green-950/40 p-4">
                <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-green-600 dark:text-green-400" />
                <span className="text-green-700 dark:text-green-300">{success}</span>
              </div>
            )}

            {/* Change Password Button */}
            <button
              type="submit"
              disabled={
                isChanging ||
                !currentPassword.trim() ||
                !newPassword.trim() ||
                !confirmPassword.trim()
              }
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-amber-500 py-4 font-semibold text-white transition-colors hover:bg-amber-600 disabled:bg-gray-300 dark:disabled:bg-muted disabled:text-gray-500 dark:disabled:text-muted-foreground"
            >
              {isChanging ? (
                <>
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  Changing Password...
                </>
              ) : (
                <>
                  <Key className="h-5 w-5" />
                  Change Password
                </>
              )}
            </button>
          </form>

          {/* Security Notice */}
          <div className="mt-6 rounded-lg border bg-amber-50 dark:bg-amber-950/40 p-4">
            <div className="text-sm">
              <p className="mb-2 font-medium text-amber-800 dark:text-amber-200">Security Notice:</p>
              <ul className="space-y-1 text-amber-700 dark:text-amber-300">
                <li>• Make sure to remember your new password</li>
                <li>• This will update the password for all wallet operations</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
