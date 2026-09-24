import {clsx, type ClassValue} from 'clsx';
import {twMerge} from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatTimeAgo(timestampMs: number): string {
  const minutes = Math.floor((Date.now() - timestampMs) / 60_000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function getErrorMessage(
  error: unknown,
  fallbackMessage = 'An unexpected error occurred'
): string {
  if (!error) return fallbackMessage;

  // Extract base error message
  let errorMessage = error instanceof Error ? error.message : String(error);

  // Remove "Error invoking remote method 'method-name':" prefix from Electron IPC errors
  if (errorMessage.includes('Error invoking remote method')) {
    const match = errorMessage.match(/Error invoking remote method[^:]*:\s*(.+)/);
    if (match && match[1]) {
      errorMessage = match[1].trim();
    }
  }

  // Optionally remove leading "Error:" prefix for cleaner messages
  errorMessage = errorMessage.replace(/^Error:\s*/i, '');

  return errorMessage || fallbackMessage;
}
