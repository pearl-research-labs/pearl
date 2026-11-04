import type {WalletSendError} from '../../../types/app-bridge';

export interface SendFailure {
  message: string;
  canRetry: boolean;
}

export function getSendFailure(error: WalletSendError): SendFailure {
  // Code -9 is the fresh-send preflight failure before signing; message text carries no retry authority.
  if (error.code === -9) {
    return {
      message:
        'No peers are connected. Nothing was signed or sent and your funds are untouched. ' +
        'Check your connection and try again.',
      canRetry: true,
    };
  }
  return {message: error.message, canRetry: false};
}
