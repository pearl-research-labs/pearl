import {Button} from '@/components/ui/button';
import {ArrowLeft} from 'lucide-react';
import {Link} from 'react-router-dom';

type HeaderProps = {
  title: string;
  onBack?: () => void;
};

export default function Header({title, onBack}: HeaderProps) {
  return (
    <div className="flex flex-shrink-0 items-center gap-4 p-8 pb-4">
      {onBack ? (
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          className="text-gray-700 dark:text-foreground/90 hover:bg-gray-100 dark:hover:bg-muted hover:text-gray-900 dark:hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
      ) : (
        <Button
          variant="ghost"
          size="sm"
          asChild
          className="text-gray-700 dark:text-foreground/90 hover:bg-gray-100 dark:hover:bg-muted hover:text-gray-900 dark:hover:text-foreground"
        >
          <Link to="/">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
      )}
      <h1 className="text-xl font-semibold text-gray-900 dark:text-foreground">{title}</h1>
    </div>
  );
}
